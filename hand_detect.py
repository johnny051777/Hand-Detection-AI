#hand detect with mediapipe

import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
# 把上面攝影機的 while loop 裡加入：
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 翻轉鏡頭畫面（讓它像鏡子）
    frame = cv2.flip(frame, 1)

    # 轉換顏色格式 BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 偵測手部
    results = hands.process(rgb_frame)

    # 如果有偵測到手
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 畫出關鍵點
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 儲存關鍵點 (x, y, z)
            keypoints = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            print("Hand Keypoints:", keypoints)

    # 顯示畫面
    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理資源
cap.release()
cv2.destroyAllWindows()
hands.close()