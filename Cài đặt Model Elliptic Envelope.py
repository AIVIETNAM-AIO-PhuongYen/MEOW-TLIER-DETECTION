# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:00:40 2026

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
import pandas as pd


#Process to using model
from data_loader import DataLoader
if __name__ == "__main__":
    # 1. Load dữ liệu
    loader = DataLoader("clean.csv")
    df = loader.load_data()
    df_clean = df.drop(columns=["is_outlier"])


def using_elliptic_envelope():

    
    X = df_clean.values

    model = EllipticEnvelope(contamination=0.05, random_state=1000) #95% confident
    y_pred = model.fit_predict(X)

    # Tính toán giới hạn đồ thị
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
    
    #Mèo bình thường thì dán nhãn 1, Mèo bệnh thì dán nhãn -
    plt.figure(figsize=(10, 7))
    plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='blue', label='Normal', alpha=0.6)
    plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c='red', label='Sick', marker='x')

    # Vẽ đường biên elip động
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), 
                         np.linspace(y_min, y_max, 500))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', linestyles='--')

    plt.title(f"Meow Sickness Detection (Samples: {len(X)})")
    plt.legend()
    plt.show()
    
    #Show ra danh sách outliers
    outlier_indices = np.where(y_pred == -1)[0]
    print("Danh sách mèo bệnh:", outlier_indices)

using_elliptic_envelope()