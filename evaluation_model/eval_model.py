import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# 載入模型
model = load_model(r"C:\AI_Hand_Project\model\hand_sign_model_add4word.h5")

# 載入資料
df = pd.read_csv(r"C:\AI_Hand_Project\dataset_csv\number0-9_add4word_1200each.csv", header=None)
data = df.iloc[:, 1:].values
labels = df.iloc[:, 0].astype(str).values

# 載入 scaler 的 mean 和 scale，並標準化
mean = np.load("scaler_mean.npy")
scale = np.load("scaler_scale.npy")
data = (data - mean) / scale

# Label encoding + one-hot
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)
num_classes = len(encoder.classes_)
labels_cat = to_categorical(labels_encoded, num_classes=num_classes)

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(data, labels_cat, test_size=0.2, random_state=42)

# 評估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"評估完成!")
