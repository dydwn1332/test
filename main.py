import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 아날로그 신호 생성 (원본 초음파 신호)
fs = 10  # 샘플링 주파수
t = np.linspace(0, 1, fs, endpoint=False)  # 1초 동안의 시간을 샘플링

# 랜덤 주파수와 진폭 설정
num_samples = len(t)
rand_freq = np.random.uniform(500, 1500, num_samples)  # 주파수를 랜덤하게 설정
rand_amplitude = np.random.uniform(0.5, 1.5, num_samples)  # 진폭을 랜덤하게 설정

ultrasound_signal = rand_amplitude * np.sin(2 * np.pi * rand_freq * t)

# 고패스 필터 설계
cutoff_frequency = 3.0  # 고패스 필터의 차단 주파수
b, a = signal.butter(4, cutoff_frequency, 'high', fs=fs, analog=False)  # 4차 Butterworth 고패스 필터

# 필터링
filtered_signal = signal.lfilter(b, a, ultrasound_signal)

# 결과 시각화
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, ultrasound_signal)
plt.title("Original Analog Ultrasound Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal, color='blue')
plt.title("High-Pass Filtered Ultrasound Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
