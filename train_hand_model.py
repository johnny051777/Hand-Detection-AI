import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# 讀取資料
df = pd.read_csv("number_0-9.csv", header=None)
data = df.iloc[:, 1:].values  # 63 維度的手部關鍵點
labels = df.iloc[:, 0].values  # 手勢數字 0~9

# 標準化資料
scaler = StandardScaler()
data = scaler.fit_transform(data)

# One-hot 編碼 labels
labels_cat = to_categorical(labels, num_classes=10)

# 資料切分
X_train, X_test, y_train, y_test = train_test_split(data, labels_cat, test_size=0.2, random_state=42)

# 建立模型
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# 儲存模型與標準化器
model.save("hand_sign_model.h5")
np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)

print("✅ 模型與 scaler 已儲存完成")
