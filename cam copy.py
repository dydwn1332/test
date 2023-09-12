import numpy as np
import matplotlib.pyplot as plt

# 샘플링 주파수와 시간 범위 설정
fs = 10000  # 샘플링 주파수 (1초에 샘플 개수)
t = np.linspace(0, 1, fs, endpoint=False)  # 1초간의 시간을 샘플링

# 초음파 신호 생성
frequency = 1000  # 초음파의 주파수 (1000 Hz)
amplitude = 1.0  # 진폭

# 정현파 생성
ultrasound_signal = amplitude * np.sin(2 * np.pi * frequency * t)

# 결과 시각화
plt.figure(figsize=(10, 4))
plt.plot(t, ultrasound_signal, 'b')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Ultrasound Signal')
plt.grid(True)
plt.show()
