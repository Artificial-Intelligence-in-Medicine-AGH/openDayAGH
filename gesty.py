import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Inicjalizacja modułów MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(p1, p2, p3):
    """Oblicza kąt między trzema punktami."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # Unikaj dzielenia przez zero

    angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def recognize_gesture(hand_landmarks, frame_shape):
    """Rozpoznaje proste gesty dłoni na podstawie landmarków."""
    h, w, _ = frame_shape
    if hand_landmarks:
        landmarks = hand_landmarks.landmark

        # Pobierz pozycje kluczowych palców
        thumb_tip = (int(landmarks[mp_hands.HandLandmark.THUMB_TIP].x * w), int(landmarks[mp_hands.HandLandmark.THUMB_TIP].y * h))
        index_tip = (int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w), int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h))
        middle_tip = (int(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * w), int(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * h))
        ring_tip = (int(landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x * w), int(landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y * h))
        pinky_tip = (int(landmarks[mp_hands.HandLandmark.PINKY_TIP].x * w), int(landmarks[mp_hands.HandLandmark.PINKY_TIP].y * h))

        thumb_mcp = (int(landmarks[mp_hands.HandLandmark.THUMB_MCP].x * w), int(landmarks[mp_hands.HandLandmark.THUMB_MCP].y * h))
        index_mcp = (int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * w), int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * h))
        middle_mcp = (int(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * w), int(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * h))
        ring_mcp = (int(landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].x * w), int(landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y * h))
        pinky_mcp = (int(landmarks[mp_hands.HandLandmark.PINKY_MCP].x * w), int(landmarks[mp_hands.HandLandmark.PINKY_MCP].y * h))

        # Kciuk w górę
        if thumb_tip[1] < thumb_mcp[1] and \
           index_tip[1] > index_mcp[1] and \
           middle_tip[1] > middle_mcp[1] and \
           ring_tip[1] > ring_mcp[1] and \
           pinky_tip[1] > pinky_mcp[1]:
            return "Thumbs up"

        # Kciuk w dół
        if thumb_tip[1] > thumb_mcp[1] and \
           index_tip[1] > index_mcp[1] and \
           middle_tip[1] > middle_mcp[1] and \
           ring_tip[1] > ring_mcp[1] and \
           pinky_tip[1] > pinky_mcp[1]:
            return "Thumbs down"

        # Środkowy palec (uproszczone)
        if index_tip[1] < index_mcp[1] and \
           middle_tip[1] < middle_mcp[1] and \
           ring_tip[1] > ring_mcp[1] and \
           pinky_tip[1] > pinky_mcp[1] and \
           thumb_tip[0] < thumb_mcp[0] + 30 and thumb_tip[0] > thumb_mcp[0] - 30:
            return "Middle fingers"

        # # Zaciśnięta pięść
        # if index_tip[1] > index_mcp[1] and \
        #    middle_tip[1] > middle_mcp[1] and \
        #    ring_tip[1] > ring_mcp[1] and \
        #    pinky_tip[1] > pinky_mcp[1] and \
        #    thumb_tip[0] > thumb_mcp[0] - 20:
        #     return "Zaciśnięta pięść"

        # V-sign
        if index_tip[1] < index_mcp[1] and \
           middle_tip[1] < middle_mcp[1] and \
           ring_tip[1] > ring_mcp[1] and \
           pinky_tip[1] > pinky_mcp[1] and \
           thumb_tip[1] < thumb_mcp[1]:
            return "V-sign"

        # OK-sign (uproszczone)
        distance_ok = ((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)**0.5
        if distance_ok < 60 and \
           middle_tip[1] < middle_mcp[1] + 10 and \
           ring_tip[1] < ring_mcp[1] + 10 and \
           pinky_tip[1] < pinky_mcp[1] + 10:
            return "OK-sign"

        # Otwarta dłoń
        if index_tip[1] < index_mcp[1] and \
           middle_tip[1] < middle_mcp[1] and \
           ring_tip[1] < ring_mcp[1] and \
           pinky_tip[1] < pinky_mcp[1] and \
           thumb_tip[1] < thumb_mcp[1]:
            return "Open hand"

        return "Calibrating..."

# Uruchomienie kamery
cap = cv2.VideoCapture(0)

# Zmienne do śledzenia ruchu dłoni
previous_hand_center_x = None
waving_threshold = 30  # Minimalna zmiana pozycji X do uznania za machanie
waving_frames = 0
min_waving_frames = 5 # Liczba klatek z ruchem do rozpoznania machania

# Zmienne do interakcji
possible_gestures = ["Thumbs up", "Thumbs down", "Middle fingers", "V-sign", "OK-sign", "Open hand"]
expected_gesture = None
instruction_time = 5  # Czas w sekundach między instrukcjami
instruction_start_time = time.time()
correct_gesture = False
feedback_duration = 2  # Czas wyświetlania informacji o poprawności
feedback_end_time = 0

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
     mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Konwersja do RGB (MediaPipe wymaga takiego formatu)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Przetwarzanie klatki
        results_hands = hands.process(frame_rgb)
        results_pose = pose.process(frame_rgb)

        # Tworzenie maski segmentacyjnej
        mask = np.zeros_like(frame)
        current_hand_center_x = None
        recognized_gesture = "Calibrating..."

        # Jeśli wykryto dłonie
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Rysowanie szkieletu dłoni
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

                # Rysowanie segmentacji dłoni
                for landmark in hand_landmarks.landmark:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(mask, (cx, cy), 20, (255, 255, 255), -1)

                # Rozpoznawanie statycznego gestu
                recognized_gesture = recognize_gesture(hand_landmarks, frame.shape)
                cv2.putText(frame, f"Recognized: {recognized_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

                # Oznaczanie indeksów palców
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.putText(frame, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1, cv2.LINE_AA)

        #         # Śledzenie ruchu dłoni dla machania
        #         hand_center_x = int(sum(landmark.x for landmark in hand_landmarks.landmark) / len(hand_landmarks.landmark) * w)
        #         if previous_hand_center_x is not None:
        #             movement = abs(hand_center_x - previous_hand_center_x)
        #             if movement > waving_threshold:
        #                 waving_frames += 1
        #             else:
        #                 waving_frames = max(0, waving_frames - 2)

        #             if waving_frames > min_waving_frames:
        #                 cv2.putText(frame, "Rozpoznano: Machanie", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        #                 recognized_gesture = "Machanie" # Aktualizujemy rozpoznany gest
        #         previous_hand_center_x = hand_center_x

        # else:
        #     previous_hand_center_x = None
        #     waving_frames = 0

        # Logika instrukcji i sprawdzania poprawności
        current_time = time.time()
        if expected_gesture is None or current_time - instruction_start_time >= instruction_time:
            expected_gesture = random.choice(possible_gestures)
            instruction_start_time = current_time
            correct_gesture = False
            feedback_end_time = 0

        cv2.putText(frame, f"Make: {expected_gesture}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if expected_gesture is not None and recognized_gesture != "Calibrating..." and not correct_gesture:
            if recognized_gesture == expected_gesture:
                correct_gesture = True
                feedback_end_time = current_time + feedback_duration
            elif current_time >= feedback_end_time and feedback_end_time > 0:
                expected_gesture = None # Resetuj po wyświetleniu feedbacku o błędzie

        if correct_gesture:
            cv2.putText(frame, "YOU DID IT!", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            if current_time >= feedback_end_time:
                expected_gesture = None # Resetuj po wyświetleniu feedbacku o poprawności
        elif expected_gesture is not None and recognized_gesture != "calibrating..." and current_time < feedback_end_time:
            cv2.putText(frame, "WHAT IS WRONG WITH YOU!", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


        # Jeśli wykryto całe ciało (bez zmian)
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2))

            # Segmentacja ciała
            for landmark in results_pose.pose_landmarks.landmark:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(mask, (cx, cy), 10, (255, 255, 255), -1)

        # Nałożenie maski na obraz (bez zmian)
        

        # Wyświetlanie obrazu
        cv2.imshow('Full Body Tracking', frame)
      

        # Wyjście po naciśnięciu 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()