import cv2
import mediapipe as mp

# Initialize Mediapipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Dictionary to map hand gestures to letters
gesture_to_letter = {
    'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E',
    'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J',
    'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O',
    'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T',
    'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z'
}

def identify_letter(landmarks):
    """
    Identify the letter based on the hand landmark positions.
    """
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Letter "A": Thumb across the palm, all fingers closed.
    if thumb_tip.x < index_tip.x and all(f.y > index_tip.y for f in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return gesture_to_letter['A']

    # Letter "B": All fingers straight up, thumb tucked across the palm.
    if (thumb_tip.y > landmarks[3].y and all(f.y < landmarks[2].y for f in [index_tip, middle_tip, ring_tip, pinky_tip])):
        return gesture_to_letter['B']

    # Letter "C": Fingers and thumb form a "C" shape.
    if (thumb_tip.x > index_tip.x and pinky_tip.x < index_tip.x and
            middle_tip.y < ring_tip.y and middle_tip.x < thumb_tip.x):
        return gesture_to_letter['C']

    # Letter "D": Index finger pointing up, other fingers bent.
    if (index_tip.y < landmarks[6].y and middle_tip.y > landmarks[10].y and
        ring_tip.y > landmarks[14].y and pinky_tip.y > landmarks[18].y):
        return gesture_to_letter['D']

    # Letter "E": All fingers curled, thumb tucked under.
    if (thumb_tip.x < index_tip.x and all(f.y > landmarks[6].y for f in [index_tip, middle_tip, ring_tip, pinky_tip])):
        return gesture_to_letter['E']

    # Letter "F": OK sign (index and thumb touch, other fingers up).
    if (abs(index_tip.x - thumb_tip.x) < 0.05 and all(f.y < landmarks[6].y for f in [middle_tip, ring_tip, pinky_tip])):
        return gesture_to_letter['F']

    # Letter "G": Thumb and index finger form an open "G" shape.
    if (thumb_tip.y < index_tip.y and middle_tip.y > index_tip.y and
            ring_tip.y > index_tip.y and pinky_tip.y > index_tip.y):
        return gesture_to_letter['G']

    # Letter "H": Index and middle fingers pointing out, thumb and other fingers tucked in.
    if (index_tip.y < landmarks[6].y and middle_tip.y < landmarks[10].y and
            ring_tip.y > landmarks[10].y and pinky_tip.y > landmarks[10].y):
        return gesture_to_letter['H']

    # Letter "I": Only pinky finger raised.
    if (pinky_tip.y < landmarks[18].y and all(f.y > landmarks[10].y for f in [thumb_tip, index_tip, middle_tip, ring_tip])):
        return gesture_to_letter['I']

    # Letter "J": Index finger pointing up, other fingers bent.
    if (index_tip.y < landmarks[6].y and middle_tip.y > landmarks[10].y and
            ring_tip.y > landmarks[14].y and pinky_tip.y > landmarks[18].y):
        return gesture_to_letter['J']

    # Letter "K": Index and middle fingers up, other fingers bent.
    if (index_tip.y < landmarks[6].y and middle_tip.y < landmarks[10].y and
            ring_tip.y > landmarks[10].y and pinky_tip.y > landmarks[18].y):
        return gesture_to_letter['K']

    # Letter "L": Thumb and index finger form an "L" shape.
    if (thumb_tip.y > index_tip.y and middle_tip.y > index_tip.y and
            ring_tip.y > index_tip.y and pinky_tip.y > index_tip.y):
        return gesture_to_letter['L']

    # Letter "M": All fingers up, thumb across the palm.
    if (thumb_tip.x < index_tip.x and all(f.y < landmarks[2].y for f in [index_tip, middle_tip, ring_tip, pinky_tip])):
        return gesture_to_letter['M']

    # Letter "N": Index and middle fingers up, other fingers bent.
    if (index_tip.y < landmarks[6].y and middle_tip.y < landmarks[10].y and
            ring_tip.y > landmarks[10].y and pinky_tip.y > landmarks[18].y):
        return gesture_to_letter['N']

    # Letter "O": Thumb and index finger form a circle, other fingers bent.
    if (abs(index_tip.x - thumb_tip.x) < 0.05 and abs(index_tip.y - thumb_tip.y) < 0.05 and
            middle_tip.y > landmarks[10].y and ring_tip.y > landmarks[14].y and pinky_tip.y > landmarks[18].y):
        return gesture_to_letter['O']

    # Letter "P": Thumb and index finger form a "P" shape, other fingers bent.
    if (thumb_tip.x > index_tip.x and thumb_tip.y < index_tip.y and
            middle_tip.y > landmarks[10].y and ring_tip.y > landmarks[14].y and pinky_tip.y > landmarks[18].y):
        return gesture_to_letter['P']

    # Letter "Q": Thumb and index finger form a "Q" shape, other fingers bent.
    if (thumb_tip.x > index_tip.x and thumb_tip.y > index_tip.y and
            middle_tip.y > landmarks[10].y and ring_tip.y > landmarks[14].y and pinky_tip.y > landmarks[18].y):
        return gesture_to_letter['Q']

    # Letter "R": Index and middle fingers up, thumb and other fingers bent.
    if (index_tip.y < landmarks[6].y and middle_tip.y < landmarks[10].y and
            thumb_tip.y > landmarks[3].y and ring_tip.y > landmarks[14].y and pinky_tip.y > landmarks[18].y):
        return gesture_to_letter['R']

    # Letter "S": All fingers and thumb curled inward.
    if (all(f.y > index_tip.y for f in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip])):
        return gesture_to_letter['S']

    # Letter "T": Index finger pointing up, other fingers and thumb bent.
    if (index_tip.y < landmarks[6].y and all(f.y > index_tip.y for f in [thumb_tip, middle_tip, ring_tip, pinky_tip])):
        return gesture_to_letter['T']

    # Letter "U": Thumb and index finger form a "U" shape, other fingers up.
    if (thumb_tip.y > index_tip.y and all(f.y < index_tip.y for f in [middle_tip, ring_tip, pinky_tip])):
        return gesture_to_letter['U']

    # Letter "V": Index and middle fingers form a "V" shape, other fingers bent.
    if (index_tip.y < landmarks[6].y and middle_tip.y < landmarks[10].y and
            ring_tip.y > landmarks[10].y and pinky_tip.y > landmarks[18].y):
        return gesture_to_letter['V']

    # Letter "W": Index, middle, and ring fingers up, thumb and pinky bent.
    if (index_tip.y < landmarks[6].y and middle_tip.y < landmarks[10].y and
            ring_tip.y < landmarks[14].y and thumb_tip.y > landmarks[3].y and pinky_tip.y > landmarks[18].y):
        return gesture_to_letter['W']

    # Letter "X": Index and middle fingers form an "X" shape, other fingers bent.
    if (index_tip.x < middle_tip.x and index_tip.y < landmarks[6].y and middle_tip.y < landmarks[10].y and
            ring_tip.y > landmarks[10].y and pinky_tip.y > landmarks[18].y):
        return gesture_to_letter['X']

    # Letter "Y": Thumb and pinky fingers up, other fingers bent.
    if (thumb_tip.y < landmarks[3].y and pinky_tip.y < landmarks[18].y and
            index_tip.y > landmarks[6].y and middle_tip.y > landmarks[10].y and ring_tip.y > landmarks[14].y):
        return gesture_to_letter['Y']

    # Letter "Z": Index finger pointing up, thumb and other fingers bent.
    if (index_tip.y < landmarks[6].y and all(f.y > index_tip.y for f in [thumb_tip, middle_tip, ring_tip, pinky_tip])):
        return gesture_to_letter['Z']

    # If no letter is identified, return '?'
    return '?'

def recognize_sign_language():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame to avoid mirrored output
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Convert BGR image to RGB for mediapipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks on the frame for visual feedback
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Identify the letter based on the landmarks
                letter = identify_letter(hand_landmarks.landmark)

                # Display the identified letter on the frame
                cv2.putText(frame, f'Letter: {letter}', (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the output
        cv2.imshow("ASL Recognition", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_sign_language()
