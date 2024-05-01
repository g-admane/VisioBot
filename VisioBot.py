import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import serial.tools.list_ports
import sys, os

'''if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app 
    # path into variable _MEIPASS'.
    application_path = sys._MEIPASS
else:
    None
    #application_path = os.path.dirname(os.path.abspath(__file__))'''

class DummySerial:
    def __init__(self, port, baudrate, timeout):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        print(f"Initialized dummy serial port: {port}")

    def write(self, data):
        print(f"Writing to dummy port: {data}")

    def close(self):
        print("Closing dummy port.")



# Function to list all available ports
def list_ports():
    ports = list(serial.tools.list_ports.comports())
    for index, p in enumerate(ports):
        print(f"Index = {index}: {p.device} - {p.description}")
    return ports if ports else None

# Function to let the user select the port
def select_port(ports):
    if ports is None:
        print("No real ports found. Using Dummy Arduino.")
        return DummySerial('COM0', 9600, timeout=1)
    while True:
        try:
            port_index = int(input("Enter the index of the Arduino port: "))
            if 0 <= port_index < len(ports):
                return ports[port_index].device
            else:
                print(f"Invalid index. Please enter a number between 0 and {len(ports)-1}.")
        except ValueError:
            print("Please enter a valid integer.")

# List available ports and ask the user to select the Arduino port
ports = list_ports()
arduino_port = select_port(ports)

if isinstance(arduino_port, DummySerial):
    arduino = arduino_port
else:
    try:
        arduino = serial.Serial(arduino_port, 9600, timeout=1)
        time.sleep(2)  # Allow time for the connection to establish
        print(f"Connected to Arduino on {arduino_port}")
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")


def send_gesture_commands(hand_landmarks, handedness, movement_direction):
    # True for Grip, False for Release
    grip_status = is_hand_open(hand_landmarks, handedness)
    movement_direction = movement_direction
    send_combined_commands(grip_status, movement_direction)


# Initialize the timers for each movement direction
movement_timers = {"U": 0, "D": 0, "L": 0, "R": 0}  # Up  # Down  # Left  # Right
start_time = None


# Function to update the timers based on detected movement directions
def update_movement_timers(movement_direction):
    for direction in movement_timers:
        if direction in movement_direction:
            movement_timers[direction] = time.time()
        else:
            # Check if the direction has been absent for more than 500 ms
            if time.time() - movement_timers[direction] > 0.5:
                movement_timers[direction] = 0


# Function to construct the command string
def send_combined_commands(grip, movement_dir):
    # Update the timers based on the detected movement directions
    update_movement_timers(movement_dir)
    # Construct the command string
    command_string = f"G{int(grip)}"
    for direction in "UDLR":
        command_string += f"{int(movement_timers[direction] > 0)}"
    command_string += "\n"
    print(command_string)  # For debugging or testing
    arduino.write(command_string.encode())


# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)


# Function to determine if the hand is open or closed
def is_hand_open(hand_landmarks, handedness):
    # Assuming thumb tip is landmark 4 and index finger tip is landmark 8
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    # Calculate the distance between the thumb tip and index tip in 3D space
    distance = np.sqrt((thumb_tip.x - index_tip.x)**2 +
                       (thumb_tip.y - index_tip.y)**2 +
                       (thumb_tip.z - index_tip.z)**2)
    open_hand_threshold = 0.07  # Adjust this threshold based on testing
    return distance > open_hand_threshold


# Function to determine the direction of hand movement
def get_hand_movement_direction(prev_landmark_px,
                                curr_landmark_px,
                                threshold=6):
    delta_x = curr_landmark_px[0] - prev_landmark_px[0]
    delta_y = curr_landmark_px[1] - prev_landmark_px[1]
    direction = ""
    if abs(delta_x) > threshold:
        direction += "Left" if delta_x < 0 else "Right"
    if abs(delta_y) > threshold:
        direction += "Up" if delta_y < 0 else "Down"
    return direction


# Capture video from the laptop's webcam
cap = cv2.VideoCapture(0)


# Variables to store the previous central landmark position
prev_landmark_x = None
prev_landmark_y = None
hand_detected_start_time = None
hand_last_seen_time = None  # Initialize the last seen timer

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    # Flip the image horizontally for a later selfie-view display
    # Convert the BGR image to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    # Convert the image color back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Draw the hand annotations on the image
    if results.multi_hand_landmarks:
        if hand_detected_start_time is None:
            hand_detected_start_time = time.time()
        hand_last_seen_time = time.time()  # Update the last seen time
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            mp_drawing.draw_landmarks(image, hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS)
            # Check if hand is open or closed and print the result on the video
            hand_status = ("Release" if is_hand_open(hand_landmarks,
                                                     handedness) else "Grip")
            cv2.putText(
                image,
                hand_status,
                (image.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            # Get the central landmark of the hand (e.g., the wrist)
            # Wrist as the central point
            curr_landmark = hand_landmarks.landmark[0]
            if prev_landmark_x is not None and prev_landmark_y is not None:
                # Convert landmark positions to relative pixel coordinates
                frame_height, frame_width, _ = image.shape
                prev_landmark_px = np.array([
                    prev_landmark_x * frame_width,
                    prev_landmark_y * frame_height
                ])
                curr_landmark_px = np.array([
                    curr_landmark.x * frame_width,
                    curr_landmark.y * frame_height
                ])
                # Determine the direction of movement
                movement_direction = get_hand_movement_direction(
                    prev_landmark_px, curr_landmark_px)
                print("movement_direction = ", movement_direction)
                # Check if the hand has been present for more than 5 seconds
                if hand_detected_start_time and (time.time() -
                                                 hand_detected_start_time > 5):
                    send_gesture_commands(hand_landmarks, handedness,
                                          movement_direction)
                cv2.putText(
                    image,
                    movement_direction,
                    (image.shape[1] - 200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                tips = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]
                palm_base = hand_landmarks.landmark[0]
                M_F = 0.167
                # Calculate the distance from each finger tip to the palm base
                distances_mf = [
                    np.sqrt((tip.x - palm_base.x)**2 +
                            (tip.y - palm_base.y)**2 +
                            (tip.z - palm_base.z)**2) for tip in tips
                ]
                # Check if the index, ring, and pinky finger tips are closer to the palm base
                # and the mf tip is far from the base
                if all(distances_mf[i] + M_F < distances_mf[2]
                       for i in [1, 3, 4]):
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time > 3:
                        cv2.putText(
                            image,
                            "Please Be Polite",
                            (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (230, 216, 173),
                            2,
                            cv2.LINE_AA,
                        ) #  ;)
                    #else:
                    #start_time = None
            # Update the previous landmark positions
            prev_landmark_x = curr_landmark.x
            prev_landmark_y = curr_landmark.y
    else:
        # Check if the hand has been absent for more than 1 second
        if hand_last_seen_time and (time.time() - hand_last_seen_time > 1):
            hand_detected_start_time = None  # Reset the timer
    cv2.namedWindow("VisioBot", cv2.WINDOW_NORMAL)
    cv2.imshow("VisioBot", image)
    if cv2.waitKey(5) & 0xFF == ord("q"or "Q"):
        break


hands.close()
cap.release()
cv2.destroyAllWindows()
arduino.close()
print("Arduino connection closed.")
