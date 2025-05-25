# train_hand_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from data_preprocess import load_and_preprocess_data
import numpy as np

# 載入與前處理資料
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(r"C:\AI_Hand_Project\dataset_csv\number_0-9.csv")


# 建立模型
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# 儲存模型與標準化參數
model.save("hand_sign_model.h5")
np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)

print("✅ 模型與 scaler 已儲存完成")
