# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:57:02 2026

@author: Admin
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


#Process to using model
from data_loader import DataLoader
if __name__ == "__main__":
    # 1. Load dữ liệu
    loader = DataLoader("clean.csv")
    df = loader.load_data()
    df_clean = df.drop(columns=["is_outlier"])



def scratch_code():
    
    import numpy as np
    loader = DataLoader("clean.csv")
    df = loader.load_data()
    df=df.drop(columns=["is_outlier"])              #Delete the columns
    
    
    data_matrix=np.array(df)                                #Changing to array for calculations
    
    n_samples,n_features= df.shape
    clean_samples= int(n_samples*(1-0.05)) #Bởi vì chỉ có 95% số được chọn 
    print(clean_samples)
    
    chosen_number=np.random.choice(n_samples #thông số lấy từ mẫu 
                                       ,clean_samples #Size
                                       ,replace=False)
    chosen_det=1000000000000000000000000000000000000000
    chosen_mean=0
    chosen_cov=0
    
    for i in range(8):
        chosen_array=data_matrix[chosen_number]
        mean = np.mean(chosen_array, axis=0)
        cov = np.cov(chosen_array, rowvar=False)
        det = np.linalg.det(cov)
        print(f"Vòng lặp {i+1}: Determinant = {det:.6f}")
        if det < chosen_det:  
            chosen_det = det
            chosen_mean = mean
            chosen_cov = cov
        all_cat = np.array([])
        inv_cov=np.linalg.inv(cov)
        for j in range(len(data_matrix)):
            d = data_matrix[j] - mean
            calculated_mahalanobis=d @ inv_cov @ d.T
            all_cat=np.append(all_cat,calculated_mahalanobis) 
        
        # Cập nhật lại chosen_number bằng cách lấy 47 con gần tâm nhất
        chosen_number = np.argsort(all_cat)[:clean_samples]
    
    print(chosen_det)
    
    
    inverse_cov_matrix= np.linalg.inv(chosen_cov)     #Hàm -1  
    outliers=[]
    for i in range(len(data_matrix)):
        d=np.array([])
        d=np.append(d, data_matrix[i]- chosen_mean)          #Calculate nornam distance
        Mahalanobis_distance= d@inverse_cov_matrix@d        #Calculate Mahalanobis (Chi-square) distance
        if Mahalanobis_distance > 7.378:  #Confidential interval 95%
            outliers.append(i)
    
    
    outliers_df = df.loc[outliers]
    print(outliers_df)
    
    
    
    
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    
    # 1. Vẽ toàn bộ các điểm màu xanh dương
    plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c='blue', label='Normal')
    
    # 2. Vẽ đè các điểm Outlier lên bằng màu đỏ
    plt.scatter(data_matrix[outliers, 0], data_matrix[outliers, 1], c='red', label='Sick')
    
    # 3. Tính toán và vẽ vòng elip bao quanh (dùng kết quả xịn nhất bạn đã tìm được)
    vals, vecs = np.linalg.eigh(chosen_cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    # 7.378 là ngưỡng Chi-square bạn đang dùng trong code
    w, h = 2 * np.sqrt(vals * 7.378)
    
    ax = plt.gca()
    ellipse = Ellipse(xy=chosen_mean, width=w, height=h, angle=theta, 
                      edgecolor='black', fc='none', lw=2, ls='--')
    ax.add_patch(ellipse)
    
    plt.xlabel('Food (gram)')
    plt.ylabel('Sleep (hours)')
    plt.legend()
    plt.show()

scratch_code() # Tự code ML dựa trên chi_square và robust 

def using_elliptic_envelope():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.covariance import EllipticEnvelope

    X = df.values

    model = EllipticEnvelope(contamination=0.05, random_state=1000)
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