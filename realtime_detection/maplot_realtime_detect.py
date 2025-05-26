import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image

# 讀取模型與標準化參數
model = load_model(r"C:\AI_Hand_Project\hand_sign_model_add4word.h5")
mean = np.load("scaler_mean.npy")
scale = np.load("scaler_scale.npy")

# Mediapipe 初始化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# 攝影機初始化
cap = cv2.VideoCapture(0)
print("🚀 啟動即時手語辨識，按 q 離開")

# 函數：將 Matplotlib 圖轉成 OpenCV 圖
def plot_prediction_bar(predictions):
    plt.clf()
    classes = list(range(10))
    bars = plt.bar(classes, predictions, color='skyblue')
    plt.ylim([0, 1])
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.title("Prediction Probabilities")

    # 設定 x 軸刻度為 0~9
    plt.xticks(classes, [str(c) for c in classes])

    # 每根長條上面顯示機率值
    for i, prob in enumerate(predictions):
        plt.text(i, prob + 0.02, f"{prob*100:.1f}%", ha='center', va='bottom', fontsize=8)

    # 顯示 Top 3 結果在圖表右上角
    top3_indices = np.argsort(predictions)[-3:][::-1]
    top3_text = "\n".join([
        f"Top{i+1}: {idx} ({predictions[idx]*100:.1f}%)"
        for i, idx in enumerate(top3_indices)
    ])
    plt.text(9.7, 1.0, top3_text, va='top', ha='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_pil = Image.open(buf)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGR)
    buf.close()
    return img_cv


plt.ion()  # 開啟即時互動模式
plt.figure(figsize=(4, 3))  # 小圖表尺寸

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    predictions = np.zeros(10)  # 預設為全0

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
                prediction = model.predict(keypoints, verbose=0)[0]
                predictions = prediction
                predicted_class = np.argmax(prediction)
                confidence = prediction[predicted_class]

                text = f"Predict: {predicted_class} ({confidence*100:.1f}%)"
                cv2.putText(frame, text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 繪製長條圖
    bar_img = plot_prediction_bar(predictions)
    bar_img = cv2.resize(bar_img, (300, 480))

    # 合併畫面（左右並排）
    frame = cv2.resize(frame, (640, 480))
    combined = np.hstack((frame, bar_img))

    cv2.imshow("Real-Time Hand Sign Recognition + Probabilities", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
