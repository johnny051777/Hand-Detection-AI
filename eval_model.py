from tensorflow.keras.models import load_model
import numpy as np

# 載入模型與資料
model = load_model("hand_sign_model.h5")


# 標準化
mean = np.load("scaler_mean.npy")
scale = np.load("scaler_scale.npy")

# 評估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"🔍 測試損失: {loss:.4f}, 測試準確率: {accuracy*100:.2f}%")
