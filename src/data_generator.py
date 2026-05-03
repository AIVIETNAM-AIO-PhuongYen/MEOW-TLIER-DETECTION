import numpy as np
import pandas as pd
import os

def generate_cat_data():
    """
    Hàm sinh dữ liệu giả lập (Synthetic Data) về hành vi sinh hoạt của mèo.
    Bao gồm 3 file: clean.csv, noisy.csv, extreme.csv
    """
    os.makedirs('data', exist_ok=True)
    np.random.seed(42) 
    
    n_records = 1000
    mean_food, std_food = 65, 10
    mean_sleep, std_sleep = 14, 2

    #CLEAN DATA
    # Tạo số liệu theo chuẩn Normal Distribution
    food_clean = np.clip(np.random.normal(mean_food, std_food, n_records), 10, 150) # lượng thức ăn từ 10 -> 150
    sleep_clean = np.clip(np.random.normal(mean_sleep, std_sleep, n_records), 1, 24) # số giờ ngủ từ 1 -> 24h
    
    # Tạo bảng và làm tròn số
    df_clean = pd.DataFrame({
        'food_weight_g': np.round(food_clean, 2),
        'sleep_hours': np.round(sleep_clean, 2)
    })

    # NOISY DATA (Sensor errors, human errors)
    # sai số cho lượng thức ăn là 5 và sai số cho giấc ngủ là 1
    noise_food = np.random.normal(0, 5, n_records)    
    noise_sleep = np.random.normal(0, 1, n_records) 
    

    df_noisy = pd.DataFrame({
        'food_weight_g': np.round(np.clip(food_clean + noise_food, 5, 180), 2),
        'sleep_hours': np.round(np.clip(sleep_clean + noise_sleep, 0, 24), 2)
    })

    # EXTREME DATA (Outliers injection)
    df_extreme = df_clean.copy()
    n_outliers = int(0.05 * n_records) # Tạo 5% số mèo bất thường
    outlier_indices = np.random.choice(n_records, n_outliers, replace=False)

    for idx in outlier_indices:
        outlier_type = np.random.choice(['anorexia', 'binge_eating', 'insomnia', 'lethargy'])

        if outlier_type == 'anorexia': # Bỏ ăn
            df_extreme.loc[idx, 'food_weight_g'] = np.round(np.random.uniform(0, 20), 2)  
        elif outlier_type == 'binge_eating': # Ăn vô độ
            df_extreme.loc[idx, 'food_weight_g'] = np.round(np.random.uniform(130, 200), 2) 
        elif outlier_type == 'insomnia': # Mất ngủ
            df_extreme.loc[idx, 'sleep_hours'] = np.round(np.random.uniform(2, 6), 2)   
        elif outlier_type == 'lethargy': # Ngủ li bì
            df_extreme.loc[idx, 'sleep_hours'] = np.round(np.random.uniform(20, 24), 2) 

    # 5. SAVE TO CSV
    df_clean.to_csv('data/clean.csv', index=False)
    df_noisy.to_csv('data/noisy.csv', index=False)
    df_extreme.to_csv('data/extreme.csv', index=False)
