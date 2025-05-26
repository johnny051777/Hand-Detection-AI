import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(csv_path, num_classes=14, test_size=0.2, random_state=42):
    """
    資料前處理:
    
    讀CSV檔案 --> 標準化資料 --> One-Hot Encoding --> split data
    
    """
    # 讀取CSV資料
    df = pd.read_csv(csv_path, header=None)
    data = df.iloc[:, 1:].values  # 21個keypints座標 (1-63)
    labels = df.iloc[:, 0].astype(str).values  # 標籤 (0)

    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)  # 轉成 0 ~ (n-1)
    num_classes = len(encoder.classes_)  # 自動偵測總類別數
    
    #df_raw = pd.DataFrame(data)
    #print("原始檔案")
    #print(df_raw.describe().T[['mean', 'std']])
    # 標準化
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    #df_raw = pd.DataFrame(data)
    #print("標準化後檔案")
    #print(df_raw.describe().T[['mean', 'std']])
    
    # One-hot 編碼
    labels_cat = to_categorical(labels_encoded, num_classes=num_classes)
    #print(labels_cat)
    
    # 訓練測試切分
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels_cat, test_size=test_size, random_state=random_state
    )
    
    #print(f"訓練資料筆數：{len(X_train)}")
    #print(f"測試資料筆數：{len(X_test)}")


    return X_train, X_test, y_train, y_test, scaler
