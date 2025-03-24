"""
using xgboost to train a model
"""
import os
import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# 数据目录
def normalize_landmarks_np(landmarks):
    """
    using shoulder center and shoulder width to normalize landmarks (NumPy implementation)
    :param landmarks: shape (33, 3) (x, y, visibility)
    :return: normalized landmarks
    """
    left_shoulder = landmarks[11, :2]  # (x, y)
    right_shoulder = landmarks[12, :2]

    # 计算肩膀中心
    shoulder_center = (left_shoulder + right_shoulder) / 2

    # 计算肩宽
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

    if shoulder_width == 0:  # 避免除 0
        return landmarks

    # 归一化 (x, y)
    landmarks[:, :2] = (landmarks[:, :2] - shoulder_center) / shoulder_width

    return landmarks


def create_datas(data_dir, output_info=False):
    """
    create datas from csv files
    :param data_dir: input data dir
    :param output_info: debug info option
    :return: datas and labels
    """
    X, y = [], []
    num_labels = len(os.listdir(data_dir))
    cnts = 0
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue  # 跳过非文件夹

        csv_path = os.path.join(label_dir, f"{label}.csv")
        if output_info:
            print(f"Processing {csv_path} ({cnts}/{num_labels})")
        if not os.path.exists(csv_path):
            continue  # 如果 csv 文件不存在，跳过

        # 读取 CSV
        df = pd.read_csv(csv_path)

        for frame, group in df.groupby("frame"):
            assert len(group) == 33, f"Expected 33 landmarks, got {len(group)}"  # 检查是否有 33 个关键点
            # 提取特征 (x, y, visibility)
            features = group[["x", "y", "visibility"]].values.reshape(-1, 3)  # (33, 3)

            if features.shape == (33, 3):  # 只保留完整数据
                features = normalize_landmarks_np(features)  # 归一化
                X.append(features.flatten())  # 展平为 (99,)
                y.append(cnts)
            else:
                print(f"Invalid shape: {features.shape}")

        cnts += 1

    # 转换为 NumPy 数组
    X = np.array(X)
    y = np.array(y)

    if output_info:
        print(f'dataset size: {X.shape}, label size: {y.shape}')
    return X, y


def get_model(data_path = r'datas/mldata',output_info = False):
    """
    get xgboost model
    :param data_path: data path
    :param output_info: debug info option
    :return: xgboost model
    """
    X, y = create_datas(data_path, output_info=output_info)
    # print(f'dataset size: {X.shape}, label size: {y.shape}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练 XGBoost 多分类模型
    model = xgb.XGBClassifier(
        objective="multi:softmax",  # 多分类
        num_class=len(os.listdir('datas/mldata')),  # 类别数量
        eval_metric="mlogloss",
        use_label_encoder=False
    )
    model.fit(X_train, y_train)

    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # print(y_test_pred)

    # 计算准确率
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    if output_info:
        print('xgboost model created')
        print(f"acc on train: {train_acc:.4f}")
        print(f"acc on test: {test_acc:.4f}")

    return model


if __name__ == "__main__":
    model = get_model(r'datas/mldata2', output_info=True)
