import cv2
import mediapipe as mp
import pandas as pd

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

data = []
label = None

print("按數字鍵 0~9 開始收集該手勢，按 s 存檔，q 離開")

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
            if label is not None:
                data.append([label] + keypoints)

    cv2.imshow("Collect", frame)
    k = cv2.waitKey(1) & 0xFF
    if ord('0') <= k <= ord('9'):
        label = int(chr(k))
        print(f"錄製手勢: {label}")
    elif k == ord('s'):
        df = pd.DataFrame(data)
        df.to_csv(f"number_0-9.csv", index=False, header=False)
        print(f"✅ 已將{label}儲存至csv檔")
    elif ord('h'): # h: happy
        label = chr(k)
        print(f"錄製手勢: happy")
    elif ord('f'): #f: find
        label = chr(k)
        print(f"錄製手勢: find") 
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
