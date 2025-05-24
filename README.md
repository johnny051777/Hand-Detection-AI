# Hand Detection AI

## 資料前處理
- `data_detction_hand.py`: 自動偵測手勢，輸入想要偵測的數字，把手勢的keypoint傳到CSV檔案

## 模型訓練
- `train_hand_model.py`: 基於LSTM訓練手勢模型，Dataset來自CSV檔案

## REAL-TIME 偵測
- `real_time_predict.py`: REAL-TIME即時辨識手勢
- `maplot_realtime_detect.py`: REAL-TIME即時辨識手勢，附贈模型預測的0-9softmax機率長條圖

## 模型評估
- `eval_model.py`: 評估模型準確度、LOSS等..
