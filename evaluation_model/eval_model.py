import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 載入模型
model = load_model(r"C:\AI_Hand_Project\model\hand_sign_model.h5")

# 載入資料
df = pd.read_csv(r"C:\AI_Hand_Project\dataset_csv\number_0-9.csv", header=None)
data = df.iloc[:, 1:].values
labels = df.iloc[:, 0].values

# 載入 scaler 的 mean 和 scale，並標準化
mean = np.load("scaler_mean.npy")
scale = np.load("scaler_scale.npy")
data = (data - mean) / scale

# One-hot 編碼
labels_cat = to_categorical(labels, num_classes=10)

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(data, labels_cat, test_size=0.2, random_state=42)

# 評估模型
loss, accuracy = model.evaluate(X_test, y_test)
print("評估完成!")
