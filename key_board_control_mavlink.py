from pymavlink import mavutil
import time
import keyboard

# ============================================================
# CONFIGURATION
# ============================================================

CONNECTION_STRING = "udp:0.0.0.0:14552"

COMMAND_RATE = 20
COMMAND_PERIOD = 1.0 / COMMAND_RATE

MOVE_VELOCITY = 0.2  # m/s when a movement key is pressed

# Takeoff altitude (meters)
TAKEOFF_ALTITUDE = 1.0

# Takeoff suppression (seconds)
takeoff_active_until = 0
TAKEOFF_SUPPRESS_TIME = 5.0

# Key mappings
KEY_FORWARD = "w"
KEY_BACKWARD = "s"
KEY_LEFT = "a"
KEY_RIGHT = "d"
KEY_UP = "i"
KEY_DOWN = "k"

KEY_TAKEOFF = "t"
KEY_LAND = "l"
KEY_ARM = "m"

# EKF Source switching keys
KEY_EKF_SRC1 = "1"
KEY_EKF_SRC2 = "2"
KEY_EKF_SRC3 = "3"

# MAVLink command
MAV_CMD_SET_EKF_SOURCE_SET = 42007

# ============================================================
# CONNECT
# ============================================================

print("Connecting to ArduPilot...")

master = mavutil.mavlink_connection(
    CONNECTION_STRING, source_system=255, source_component=190
)

print("Waiting for heartbeat...")
master.wait_heartbeat()

print("Connected!")
print(f"System ID: {master.target_system}")
print(f"Component ID: {master.target_component}")
print(f"Current mode: {master.flightmode}")


# ============================================================
# VELOCITY COMMAND
# ============================================================


def send_velocity(vx, vy, vz):
    """
    Send velocity command in BODY_OFFSET_NED frame.

    vx: forward (+), backward (-)
    vy: right (+), left (-)
    vz: down (+), up (-)

    Type mask 3527:
    - Ignore position
    - Use velocity
    - Ignore acceleration
    - Ignore yaw
    - Ignore yaw rate
    """

    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        3527,
        0,
        0,
        0,
        vx,
        vy,
        vz,
        0,
        0,
        0,
        0,
        0,
    )


# ============================================================
# TAKEOFF COMMAND
# ============================================================


def send_takeoff(altitude):
    """
    Send NAV_TAKEOFF command.

    This function DOES NOT wait for COMMAND_ACK.
    The main control loop therefore continues running.
    """

    print(f"\nTakeoff command sent. Target altitude: {altitude} m")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0,  # param1: pitch
        0,  # param2
        0,  # param3
        0,  # param4: yaw
        0,  # param5: latitude
        0,  # param6: longitude
        altitude,
    )

    pending_commands[mavutil.mavlink.MAV_CMD_NAV_TAKEOFF] = "Takeoff"


# ============================================================
# LAND COMMAND
# ============================================================


def send_land():
    """
    Send NAV_LAND command.

    This function DOES NOT wait for COMMAND_ACK.
    """

    print("\nLand command sent.")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0,
        0,  # param1
        0,  # param2
        0,  # param3
        0,  # param4
        0,  # param5: current latitude
        0,  # param6: current longitude
        0,  # param7
    )

    pending_commands[mavutil.mavlink.MAV_CMD_NAV_LAND] = "Land"


# ============================================================
# ARM COMMAND
# ============================================================


def send_arm():
    """
    Send COMPONENT_ARM_DISARM command to arm the vehicle.

    This function DOES NOT wait for COMMAND_ACK.
    """

    print("\nArm command sent.")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,  # param1: 1 = arm, 0 = disarm
        0,  # param2: 0 = normal, 21196 = force arm
        0,
        0,
        0,
        0,
        0,
    )

    pending_commands[mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM] = "Arm"


# ============================================================
# EKF SOURCE SWITCHING
# ============================================================


def set_ekf_source(source_set):
    """
    Switch the active EKF source set.

    1 -> EK3 source set 1
    2 -> EK3 source set 2
    3 -> EK3 source set 3

    The actual sensors assigned to each source set are
    configured by the ArduPilot EK3_SRCx parameters.
    """

    if source_set not in [1, 2, 3]:
        print(f"Invalid source set: {source_set}")
        return

    print(f"\nSwitching to EKF source set {source_set}...")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        MAV_CMD_SET_EKF_SOURCE_SET,
        0,
        source_set,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    pending_commands[MAV_CMD_SET_EKF_SOURCE_SET] = f"EKF Source Set {source_set}"


# ============================================================
# COMMAND ACK HANDLING
# ============================================================

pending_commands = {}


def process_mavlink_messages():
    """
    Process incoming MAVLink messages without blocking.

    This handles:
    - COMMAND_ACK
    - HEARTBEAT

    Nothing here can pause the main control loop.
    """

    while True:
        msg = master.recv_match(blocking=False)

        if msg is None:
            break

        msg_type = msg.get_type()

        # --------------------------------------------
        # COMMAND ACK
        # --------------------------------------------

        if msg_type == "COMMAND_ACK":

            command = msg.command

            if command in pending_commands:

                command_name = pending_commands.pop(command)

                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    print(f"\n{command_name} ACCEPTED")

                else:
                    print(f"\n{command_name} REJECTED " f"(result: {msg.result})")

        # --------------------------------------------
        # HEARTBEAT
        # --------------------------------------------

        elif msg_type == "HEARTBEAT":

            # pymavlink updates master.flightmode when
            # HEARTBEAT messages are received.
            pass


# ============================================================
# WAIT FOR GUIDED
# ============================================================

print()
print("=" * 50)
print("WAITING FOR GUIDED MODE")
print("=" * 50)

print("Use your RC switch to change LOITER → GUIDED.")
print("The Python script will NOT change the flight mode.")
print()

while True:

    master.recv_match(type="HEARTBEAT", blocking=True)

    current_mode = master.flightmode

    print(f"\rCurrent mode: {current_mode}    ", end="", flush=True)

    if current_mode == "GUIDED":
        break

    time.sleep(0.1)

print()
print()
print("GUIDED detected!")


# ============================================================
# ZERO VELOCITY / HOVER PHASE
# ============================================================

print("Sending zero velocity (hover) for 2 seconds...")

start_time = time.monotonic()

while time.monotonic() - start_time < 2.0:

    send_velocity(0.0, 0.0, 0.0)

    # Process incoming messages without blocking
    process_mavlink_messages()

    time.sleep(COMMAND_PERIOD)


# ============================================================
# KEYBOARD CONTROL LOOP
# ============================================================

print()
print("=" * 50)
print("KEYBOARD CONTROL ACTIVE")
print("=" * 50)

print("--- Movement ---")
print(f"W: Forward  ({MOVE_VELOCITY} m/s)")
print(f"S: Backward ({MOVE_VELOCITY} m/s)")
print(f"A: Left     ({MOVE_VELOCITY} m/s)")
print(f"D: Right    ({MOVE_VELOCITY} m/s)")
print(f"I: Up       ({MOVE_VELOCITY} m/s)")
print(f"K: Down     ({MOVE_VELOCITY} m/s)")
print()

print("--- Commands ---")
print(f"T: Takeoff  ({TAKEOFF_ALTITUDE} m)")
print("L: Land")
print("M: Arm")
print()

print("--- EKF Source ---")
print("1: EKF Source Set 1")
print("2: EKF Source Set 2")
print("3: EKF Source Set 3")
print()

print("Q or ESC: Exit program")
print()
print("Your RC switch is the emergency control.")
print("Switch GUIDED → LOITER if anything is wrong.")
print()


# ============================================================
# KEY STATE TRACKING
# ============================================================

prev_keys = {
    KEY_TAKEOFF: False,
    KEY_LAND: False,
    KEY_ARM: False,
    KEY_EKF_SRC1: False,
    KEY_EKF_SRC2: False,
    KEY_EKF_SRC3: False,
}

running = True

last_mode_check = time.monotonic()


# ============================================================
# MAIN CONTROL LOOP
# ============================================================

while running:

    loop_start = time.monotonic()

    # --------------------------------------------------------
    # READ KEYBOARD
    # --------------------------------------------------------

    pressed = {
        KEY_FORWARD: keyboard.is_pressed(KEY_FORWARD),
        KEY_BACKWARD: keyboard.is_pressed(KEY_BACKWARD),
        KEY_LEFT: keyboard.is_pressed(KEY_LEFT),
        KEY_RIGHT: keyboard.is_pressed(KEY_RIGHT),
        KEY_UP: keyboard.is_pressed(KEY_UP),
        KEY_DOWN: keyboard.is_pressed(KEY_DOWN),
        KEY_TAKEOFF: keyboard.is_pressed(KEY_TAKEOFF),
        KEY_LAND: keyboard.is_pressed(KEY_LAND),
        KEY_ARM: keyboard.is_pressed(KEY_ARM),
        KEY_EKF_SRC1: keyboard.is_pressed(KEY_EKF_SRC1),
        KEY_EKF_SRC2: keyboard.is_pressed(KEY_EKF_SRC2),
        KEY_EKF_SRC3: keyboard.is_pressed(KEY_EKF_SRC3),
        "q": keyboard.is_pressed("q"),
        "esc": keyboard.is_pressed("esc"),
    }

    # --------------------------------------------------------
    # CALCULATE VELOCITY
    # --------------------------------------------------------

    vx = 0.0
    vy = 0.0
    vz = 0.0

    # Forward / Backward

    if pressed[KEY_FORWARD]:
        vx += MOVE_VELOCITY

    if pressed[KEY_BACKWARD]:
        vx -= MOVE_VELOCITY

    # Right / Left

    if pressed[KEY_RIGHT]:
        vy += MOVE_VELOCITY

    if pressed[KEY_LEFT]:
        vy -= MOVE_VELOCITY

    # Up / Down

    if pressed[KEY_UP]:
        vz -= MOVE_VELOCITY

    if pressed[KEY_DOWN]:
        vz += MOVE_VELOCITY

    # --------------------------------------------------------
    # SEND VELOCITY
    # --------------------------------------------------------

    if time.monotonic() > takeoff_active_until:

        send_velocity(vx, vy, vz)

    # --------------------------------------------------------
    # TAKEOFF
    # --------------------------------------------------------

    if pressed[KEY_TAKEOFF] and not prev_keys[KEY_TAKEOFF]:

        send_takeoff(TAKEOFF_ALTITUDE)
        takeoff_active_until = time.monotonic() + TAKEOFF_SUPPRESS_TIME

    # --------------------------------------------------------
    # LAND
    # --------------------------------------------------------

    if pressed[KEY_LAND] and not prev_keys[KEY_LAND]:

        send_land()

    # --------------------------------------------------------
    # ARM
    # --------------------------------------------------------

    if pressed[KEY_ARM] and not prev_keys[KEY_ARM]:

        send_arm()

    # --------------------------------------------------------
    # EKF SOURCE SET 1
    # --------------------------------------------------------

    if pressed[KEY_EKF_SRC1] and not prev_keys[KEY_EKF_SRC1]:

        set_ekf_source(1)

    # --------------------------------------------------------
    # EKF SOURCE SET 2
    # --------------------------------------------------------

    if pressed[KEY_EKF_SRC2] and not prev_keys[KEY_EKF_SRC2]:

        set_ekf_source(2)

    # --------------------------------------------------------
    # EKF SOURCE SET 3
    # --------------------------------------------------------

    if pressed[KEY_EKF_SRC3] and not prev_keys[KEY_EKF_SRC3]:

        set_ekf_source(3)

    # --------------------------------------------------------
    # UPDATE PREVIOUS KEY STATES
    # --------------------------------------------------------

    prev_keys[KEY_TAKEOFF] = pressed[KEY_TAKEOFF]
    prev_keys[KEY_LAND] = pressed[KEY_LAND]
    prev_keys[KEY_ARM] = pressed[KEY_ARM]

    prev_keys[KEY_EKF_SRC1] = pressed[KEY_EKF_SRC1]
    prev_keys[KEY_EKF_SRC2] = pressed[KEY_EKF_SRC2]
    prev_keys[KEY_EKF_SRC3] = pressed[KEY_EKF_SRC3]

    # --------------------------------------------------------
    # PROCESS MAVLINK MESSAGES
    # --------------------------------------------------------

    process_mavlink_messages()

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if pressed["q"] or pressed["esc"]:

        print()
        print("Exit key pressed.")

        running = False
        break

    # --------------------------------------------------------
    # FLIGHT MODE SAFETY CHECK
    # --------------------------------------------------------

    current_time = time.monotonic()

    if current_time - last_mode_check >= 0.5:

        last_mode_check = current_time

        current_mode = master.flightmode

        if current_mode not in ["GUIDED", "LAND"]:

            print()
            print(f"⚠️ Flight mode changed to " f"{current_mode}! Stopping.")

            running = False
            break

    # --------------------------------------------------------
    # MAINTAIN 20 Hz LOOP
    # --------------------------------------------------------

    elapsed = time.monotonic() - loop_start

    remaining = COMMAND_PERIOD - elapsed

    if remaining > 0:
        time.sleep(remaining)


# ============================================================
# STOP
# ============================================================

print()
print("Stopping...")

# Send zero velocity for 1 second
# to make sure the last velocity command is zero.

for _ in range(20):

    send_velocity(0.0, 0.0, 0.0)

    process_mavlink_messages()

    time.sleep(COMMAND_PERIOD)

print("Velocity commands stopped.")
print("Use RC switch to return to LOITER or another mode.")
