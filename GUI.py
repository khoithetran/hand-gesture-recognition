import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import pyautogui

def run_model():
    model = tf.keras.models.load_model(r"hand-gesture-recognition\model\keypoint_classifier_model.h5")
    with open(r"hand-gesture-recognition\model\label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    label_to_key = {'up': 'up', 'down': 'down', 'left': 'left', 'right': 'right', 'normal': None}
    cap = cv2.VideoCapture(0)
    last_action = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]
                if len(landmarks) == 42:
                    prediction = model.predict(np.array(landmarks).reshape(1, -1))
                    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(frame, predicted_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                    key_to_press = label_to_key.get(predicted_label)
                    if key_to_press and key_to_press != last_action:
                        pyautogui.press(key_to_press)
                        last_action = key_to_press
                    elif key_to_press is None:
                        last_action = None

        cv2.imshow("Hand Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_model()
