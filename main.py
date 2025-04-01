import cv2
import mediapipe as mp
import numpy as np

# Inicjalizacja modułów MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Uruchomienie kamery
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
     mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Konwersja do RGB (MediaPipe wymaga takiego formatu)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Przetwarzanie klatki
        results_hands = hands.process(frame_rgb)
        results_pose = pose.process(frame_rgb)
        
        # Tworzenie maski segmentacyjnej
        mask = np.zeros_like(frame)
        
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
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(mask, (cx, cy), 20, (255, 255, 255), -1)
        
        # Jeśli wykryto całe ciało
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2))
            
            # Segmentacja ciała
            for landmark in results_pose.pose_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(mask, (cx, cy), 10, (255, 255, 255), -1)
        
        # Nałożenie maski na obraz
        segmented_body = cv2.bitwise_and(frame, mask)
        
        # Wyświetlanie obrazu
        cv2.imshow('Full Body Tracking', frame)
        cv2.imshow('Segmented Body', segmented_body)
        
        # Wyjście po naciśnięciu 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
