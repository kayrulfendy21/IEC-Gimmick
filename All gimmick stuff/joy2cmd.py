#!/usr/bin/env python3
"""
Joystick to cmd_vel converter with emergency stop.

Button mappings (typical Xbox/PS controller):
- Left stick: movement control
- Button 0 (A/X): Emergency stop toggle
- Button 1 (B/O): Clear E-stop in standalone/local mode

Emergency stop can also be triggered via /emergency_stop topic.
"""

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool


class Joy2CmdNode(Node):
    def __init__(self):
        super().__init__('joy2cmd')

        # ---------------------------------------------------------------------
        # TUNING PARAMETERS (adjust these first to reduce jerkiness)
        # ---------------------------------------------------------------------
        # Max commanded speed from full joystick deflection.
        # Reduce these for gentler manual driving.
        self.declare_parameter('max_linear_speed', 0.35)    # m/s (try 0.25 ~ 0.40)
        self.declare_parameter('max_angular_speed', 0.70)   # rad/s (try 0.50 ~ 0.90)

        # Ignore tiny joystick noise around center.
        self.declare_parameter('deadzone', 0.12)            # try 0.08 ~ 0.15

        # Rate limits (slew-rate): how fast cmd_vel is allowed to change.
        # Lower = smoother/slower response. Higher = snappier/faster response.
        self.declare_parameter('linear_accel_limit', 0.60)   # m/s^2 (try 0.30 ~ 1.00)
        self.declare_parameter('angular_accel_limit', 1.20)  # rad/s^2 (try 0.80 ~ 2.00)

        self.declare_parameter('allow_estop_clear', True)

        # Emergency stop state
        self.emergency_stopped = False
        self.localization_recovery_active = False

        # Subscribe to the /joy topic
        self.subscription = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10)

        # Subscribe to emergency stop topic (can be triggered by Nav2 or other nodes)
        self.stop_sub = self.create_subscription(
            Bool,
            'emergency_stop',
            self.emergency_stop_callback,
            10)

        # Publisher for /cmd_vel topic
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

        # Publisher for emergency stop status
        self.stop_status_pub = self.create_publisher(Bool, 'emergency_stop_active', 10)
        self.active_pub = self.create_publisher(
            Bool, '/operator/local_joystick_active', 10
        )

        recovery_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.recovery_sub = self.create_subscription(
            Bool,
            '/localization/recovery_active',
            self.recovery_active_callback,
            recovery_qos,
        )

        # Command the base controller's latched emergency-stop input. Publishing
        # only a zero Twist is not sufficient because Nav2 may publish another
        # velocity command immediately afterwards.
        self.estop_pub = self.create_publisher(Bool, 'emergency_stop', 10)

        # Adjustable max speeds
        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)

        # Input filtering / smoothing params
        self.deadzone = float(self.get_parameter('deadzone').value)
        self.linear_accel_limit = float(self.get_parameter('linear_accel_limit').value)
        self.angular_accel_limit = float(self.get_parameter('angular_accel_limit').value)

        self.allow_estop_clear = bool(self.get_parameter('allow_estop_clear').value)

        # Button indices (adjust for your controller)
        self.ESTOP_BUTTON = 0       # A button - emergency stop
        self.RESUME_BUTTON = 1      # B button - resume

        # Previous button states for edge detection
        self.prev_buttons = []

        # Rate limiter state (last commanded values)
        self.last_linear_cmd = 0.0
        self.last_angular_cmd = 0.0
        self.last_cmd_time = self.get_clock().now()

        self.get_logger().info('Joy2Cmd initialized with emergency stop')
        self.get_logger().info(f'  max_linear_speed: {self.max_linear_speed:.3f} m/s')
        self.get_logger().info(f'  max_angular_speed: {self.max_angular_speed:.3f} rad/s')
        self.get_logger().info(f'  deadzone: {self.deadzone:.3f}')
        self.get_logger().info(f'  linear_accel_limit: {self.linear_accel_limit:.3f} m/s^2')
        self.get_logger().info(f'  angular_accel_limit: {self.angular_accel_limit:.3f} rad/s^2')
        self.get_logger().info('  Press A/Button0 for emergency stop')
        if self.allow_estop_clear:
            self.get_logger().info('  Press B/Button1 to clear emergency stop')
        else:
            self.get_logger().info('  Clear E-stop from the operator UI')
        self.get_logger().info('  Left stick controls movement (no deadman button)')

    def reset_rate_limiter(self):
        """Reset smoothed command state to zero to avoid jump on resume."""
        self.last_linear_cmd = 0.0
        self.last_angular_cmd = 0.0
        self.last_cmd_time = self.get_clock().now()

    def apply_rate_limit(self, target_linear: float, target_angular: float):
        """Slew-rate limit for linear and angular commands."""
        now = self.get_clock().now()
        dt = (now - self.last_cmd_time).nanoseconds * 1e-9
        if dt <= 0.0:
            # Fallback for unusual clock behavior
            dt = 0.02  # ~50 Hz equivalent

        max_dlin = self.linear_accel_limit * dt
        max_dang = self.angular_accel_limit * dt

        dlin = target_linear - self.last_linear_cmd
        dang = target_angular - self.last_angular_cmd

        # Clamp linear delta
        if dlin > max_dlin:
            dlin = max_dlin
        elif dlin < -max_dlin:
            dlin = -max_dlin

        # Clamp angular delta
        if dang > max_dang:
            dang = max_dang
        elif dang < -max_dang:
            dang = -max_dang

        self.last_linear_cmd += dlin
        self.last_angular_cmd += dang
        self.last_cmd_time = now

        return self.last_linear_cmd, self.last_angular_cmd

    def recovery_active_callback(self, msg: Bool):
        """Yield cmd_vel while automatic localization recovery is rotating."""
        self.localization_recovery_active = msg.data
        self.active_pub.publish(Bool(data=False))

        # Reset limiter while we are not in control, so hand-back is smooth
        if self.localization_recovery_active:
            self.reset_rate_limiter()

    def emergency_stop_callback(self, msg: Bool):
        """Handle external emergency stop requests."""
        if msg.data and not self.emergency_stopped:
            self.emergency_stopped = True
            self.get_logger().warn('EMERGENCY STOP activated via topic!')
            self.reset_rate_limiter()
            self.publish_stop()
        elif not msg.data and self.emergency_stopped:
            self.emergency_stopped = False
            self.reset_rate_limiter()
            self.get_logger().info('Emergency stop cleared via topic')

    def publish_stop(self):
        """Publish zero velocity command."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher_.publish(twist)

        # Publish stop status
        status = Bool()
        status.data = self.emergency_stopped
        self.stop_status_pub.publish(status)

    def joy_callback(self, msg: Joy):
        # Initialize previous buttons if needed
        if not self.prev_buttons:
            self.prev_buttons = [0] * len(msg.buttons) if msg.buttons else []

        # Check for button presses (edge detection)
        def button_pressed(idx):
            if idx < len(msg.buttons) and idx < len(self.prev_buttons):
                return msg.buttons[idx] == 1 and self.prev_buttons[idx] == 0
            return False

        # Emergency stop button (toggle on)
        if button_pressed(self.ESTOP_BUTTON):
            self.emergency_stopped = True
            self.get_logger().warn('EMERGENCY STOP activated! Press B to resume.')
            self.estop_pub.publish(Bool(data=True))
            self.reset_rate_limiter()
            self.publish_stop()

        # Do not clear the shared stop from a raw joystick button. Clearing is
        # deliberately centralized in the operator backend and requires the
        # active control lease plus an explicit confirmation.
        if button_pressed(self.RESUME_BUTTON):
            if self.allow_estop_clear and self.emergency_stopped:
                self.emergency_stopped = False
                self.estop_pub.publish(Bool(data=False))
                self.reset_rate_limiter()
                self.get_logger().info('Emergency stop cleared')
            else:
                self.get_logger().warn('Clear E-stop from the operator UI')

        # Update previous buttons
        self.prev_buttons = list(msg.buttons) if msg.buttons else []

        # If emergency stopped, always publish zero
        if self.emergency_stopped:
            self.active_pub.publish(Bool(data=False))
            self.publish_stop()
            return

        # The recovery node owns cmd_vel while it performs its controlled spin.
        # Do not publish even a zero Twist here, because that would compete with
        # the recovery command at the base controller.
        if self.localization_recovery_active:
            self.active_pub.publish(Bool(data=False))
            self.reset_rate_limiter()
            return

        # Axis mapping for Xbox 360 controller:
        # axis 0: left stick X (steering/rotation)
        # axis 1: left stick Y (forward/backward)
        linear_input = 0.0
        angular_input = 0.0

        if len(msg.axes) > 1:
            linear_input = msg.axes[1]  # Left stick Y
        if len(msg.axes) > 0:
            angular_input = msg.axes[0]  # Left stick X

        # Deadzone: ignore small inputs to prevent drift
        if abs(linear_input) < self.deadzone:
            linear_input = 0.0
        if abs(angular_input) < self.deadzone:
            angular_input = 0.0

        # Target command from joystick
        target_linear = linear_input * self.max_linear_speed
        target_angular = angular_input * self.max_angular_speed

        # Slew-rate limited output command
        cmd_linear, cmd_angular = self.apply_rate_limit(target_linear, target_angular)

        twist = Twist()
        twist.linear.x = cmd_linear
        twist.angular.z = cmd_angular

        self.active_pub.publish(Bool(data=(linear_input != 0.0 or angular_input != 0.0)))

        # Publish the Twist message
        self.publisher_.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = Joy2CmdNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()