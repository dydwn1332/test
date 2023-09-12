import numpy as np
import matplotlib.pyplot as plt

# 가상의 초음파 데이터 생성 (랜덤 데이터)
fs = 20  # 샘플링 주파수
duration = 3  # 데이터 측정 시간 (초)
t = np.arange(0, duration, 1/fs)  # 시간 배열 생성
ultrasound_signal = np.random.uniform(-1, 1, len(t))  # 가상의 초음파 데이터

# Noise 제거
def remove_noise(signal, threshold):
    return signal[np.abs(signal) > threshold]

threshold_noise = 0.2  # 임계값 설정
ultrasound_signal_noise_removed = remove_noise(ultrasound_signal, threshold_noise)

# 연관성 없는 데이터 제거
def remove_outliers(signal, num_std):
    mean = np.mean(signal)
    std = np.std(signal)
    return signal[np.abs(signal - mean) < num_std * std]

num_std_outliers = 1.2  # 표준편차의 몇 배까지 허용할지 설정
ultrasound_signal_outliers_removed = remove_outliers(ultrasound_signal_noise_removed, num_std_outliers)

# 누락된 데이터에 특정값 지정
missing_indices = np.random.choice(len(ultrasound_signal_outliers_removed), size=10, replace=False)
missing_value = 0
ultrasound_signal_missing_imputed = ultrasound_signal_outliers_removed.copy()
ultrasound_signal_missing_imputed[missing_indices] = missing_value

# 변수 범위 설정
min_value = -0.5
max_value = 0.5
ultrasound_signal_range_set = np.clip(ultrasound_signal_missing_imputed, min_value, max_value)

# Custom 변환 작업 (예시: 로그 변환)
ultrasound_signal_transformed = np.log(ultrasound_signal_range_set + 1)

# 데이터 범위 설정
scaled_min = 0
scaled_max = 1
ultrasound_signal_scaled = (ultrasound_signal_transformed - np.min(ultrasound_signal_transformed)) / \
                            (np.max(ultrasound_signal_transformed) - np.min(ultrasound_signal_transformed)) * \
                            (scaled_max - scaled_min) + scaled_min

plt.figure(figsize=(12, 10))

plt.subplot(4, 2, 1)
plt.plot(t, ultrasound_signal, color='blue')
plt.title("Original Ultrasound Data")

plt.subplot(4, 2, 2)
plt.plot(t, ultrasound_signal_noise_removed[:len(t)], color='orange')
plt.title("After Noise Removal")

plt.subplot(4, 2, 3)
plt.plot(t, ultrasound_signal_outliers_removed[:len(t)], color='green')
plt.title("After Outlier Removal")

plt.subplot(4, 2, 4)
plt.plot(t, ultrasound_signal_missing_imputed[:len(t)], color='red')
plt.title("After Missing Data Imputation")

plt.subplot(4, 2, 5)
plt.plot(t, ultrasound_signal_range_set[:len(t)], color='purple')
plt.title("After Variable Range Setting")

plt.subplot(4, 2, 6)
plt.plot(t, ultrasound_signal_transformed[:len(t)], color='pink')
plt.title("After Custom Transformation")

plt.subplot(4, 2, 7)
plt.plot(t, ultrasound_signal_scaled[:len(t)], color='gray')
plt.title("Final Scaled Data")

plt.tight_layout()
plt.show()

