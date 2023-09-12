import numpy as np
import matplotlib.pyplot as plt

# 아날로그 신호 생성 (원본 초음파 신호)
fs = 100  # 샘플링 주파수
t = np.linspace(0, 1, fs, endpoint=False)  # 1초 동안의 시간을 샘플링

# 랜덤 주파수와 진폭 설정
num_samples = len(t)
rand_freq = np.random.uniform(500, 1500, num_samples)  # 주파수를 랜덤하게 설정
rand_amplitude = np.random.uniform(0.5, 1.5, num_samples)  # 진폭을 랜덤하게 설정

ultrasound_signal = rand_amplitude * np.sin(2 * np.pi * rand_freq * t)

# 샘플링 및 양자화
num_bits = 3  # 비트 수 (양자화 비트 수)
quantization_levels = 2 ** num_bits  # 양자화 레벨 수
quantization_step = (2 * np.max(rand_amplitude)) / quantization_levels
quantized_signal = np.round(ultrasound_signal / quantization_step) * quantization_step

# 결과 시각화
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, ultrasound_signal)
plt.title("Analog Ultrasound Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.step(t, quantized_signal, color='orange', where='mid')
plt.title("Quantized Ultrasound Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
