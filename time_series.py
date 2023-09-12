import numpy as np
import matplotlib.pyplot as plt

# 디지털 초음파 데이터 생성 (가상 데이터)
fs = 20000  # 샘플링 주파수 (Hz)
duration = 0.001  # 초음파 측정 시간 (초)
t = np.arange(0, duration, 1/fs)  # 시간 배열 생성

# 디지털 초음파 신호 생성 (가상 데이터, 여기서는 랜덤 생성)
digital_ultrasound_signal = np.random.uniform(-1, 1, len(t))

# 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(t, digital_ultrasound_signal, color='blue')
plt.title("Digital Ultrasound Time Series Data")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
