import cv2
import mediapipe as mp
import numpy as np

# Inicjalizacja modułów MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


# Uruchomienie kamery
cap = cv2.VideoCapture(0)

#punkty liczą się od góry tj. brew ma mniejszą wartość od oka
def are_eyebrows_raised(landmarks):
    left_brow_y = landmarks[70][1]
    left_eye_y = landmarks[159][1]
    
    right_brow_y = landmarks[300][1]
    right_eye_y = landmarks[386][1]

    left_diff = left_eye_y - left_brow_y
    right_diff = right_eye_y - right_brow_y

    # próg eksperymentalny - pewnie lepiej będzie dać jakąś kalibrację czy cuś (co się będzie tyczyć też wszystkich innych wyrazów twarzy)
    return (left_diff + right_diff)/2 > 10

#do kalibracji
def is_smiling(landmarks): 
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    #top_lip ma nr 13 jakby się jednak przydało
    bottom_lip = landmarks[14]

    return bottom_lip[1] > left_mouth[1] and bottom_lip[1] > right_mouth[1]

#do kalibracji
def is_frowning(landmarks):
    left_brow_y = landmarks[70][1]
    left_eye_y = landmarks[159][1]
    
    right_brow_y = landmarks[300][1]
    right_eye_y = landmarks[386][1]

    left_diff = left_eye_y - left_brow_y
    right_diff = right_eye_y - right_brow_y

    print(f"Avg diff: {(left_diff + right_diff)/2}") 

    return (left_diff + right_diff)/2 <= 9

#do kalibracji
def is_sad(landmarks):
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    bottom_lip = landmarks[14]

    return bottom_lip[1] < left_mouth[1] and bottom_lip[1] < right_mouth[1]

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
    mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh, \
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
        results_face = face_mesh.process(frame_rgb)
        
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

        #Jeśli wykryto twarz
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1))
                
                # Segmentacja twarzy
                for landmark in face_landmarks.landmark:
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(mask, (cx, cy), 5, (255, 255, 255), -1)

                    # Sprawdzanie mimiki twarzy
                    face_landmark_points = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]
                    raised_eyebrows = are_eyebrows_raised(face_landmark_points)
                    smiling = is_smiling(face_landmark_points)
                    frowning = is_frowning(face_landmark_points)
                    sad = is_sad(face_landmark_points)
                    
                    # Sprawdzanie czy dany wyraz twarzy występuje
                    if raised_eyebrows:
                        cv2.putText(frame, "UNIESIONE BRWI", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
                    if smiling:
                        cv2.putText(frame, "UŚMIECH", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
                    if frowning:
                        cv2.putText(frame, "ZMARSZCZONE BRWI", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
                    if sad:
                        cv2.putText(frame, "SMUTEK", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)

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