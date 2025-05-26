import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# 讀取模型與標準化參數
model = load_model(r"C:\AI_Hand_Project\model\hand_sign_model_add4word.h5")
mean = np.load("scaler_mean.npy")
scale = np.load("scaler_scale.npy")

# Mediapipe 初始化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# 攝影機初始化
cap = cv2.VideoCapture(0)

label_map={
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "find",
    11: "hello",
    12: "i",
    13: "phone"
}

print("🚀 啟動即時手語辨識，按 q 離開")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            if len(keypoints) == 63:
                # 標準化
                keypoints = np.array(keypoints)
                keypoints = (keypoints - mean) / scale
                keypoints = keypoints.reshape(1, -1)

                # 預測
                prediction = model.predict(keypoints)
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class]

                # 顯示預測結果
                label_txt=  label_map.get(predicted_class, str(predicted_class))
                text = f" LSTM Predict: {label_txt} ({confidence*100:.1f}%)"
                cv2.putText(frame, text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-Time Hand Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
