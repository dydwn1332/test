import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# 가상의 초음파 신호 데이터 생성
fs = 10000  # 샘플링 주파수
t = np.linspace(0, 1, fs, endpoint=False)  # 시간 벡터
f = 1000.0  # 생성할 신호의 주파수
signal = np.sin(2 * np.pi * f * t)

# 잡음 추가
noise = np.random.normal(0, 0.5, signal.shape)
noisy_signal = signal + noise

# Low-pass 필터링을 위한 함수 정의
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# 신호 필터링
cutoff_freq = 200.0  # 저주파 필터링
filtered_signal = butter_lowpass_filter(noisy_signal, cutoff_freq, fs, order=6)

# 시각화
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t, signal, 'g', label='원본 신호')
plt.title('Original Signal')
plt.subplot(3, 1, 2)
plt.plot(t, noisy_signal, 'b', label='잡음 추가된 신호')
plt.title('Noisy Signal')
plt.subplot(3, 1, 3)
plt.plot(t, filtered_signal, 'r', label='필터링된 신호')
plt.title('Filtered Signal')
plt.tight_layout()
plt.show()

# 원본 신호 (Original Signal):
# 초음파 신호의 생성에 사용한 주파수는 1000 Hz입니다.
# 초음파 신호는 정현파(sinewave) 형태를 갖습니다.
# 시간 도메인에서 주기성을 가지며, 주파수는 1000 Hz로 설정되었습니다.
# 잡음 추가된 신호 (Noisy Signal):
# 가우시안 분포를 따르는 잡음(noise)을 생성하여 원본 신호에 더해줍니다.
# 잡음은 시간에 따라 변하는 무작위한 값을 추가하므로 신호에 무작위한 변동이 생깁니다.
# 결과적으로, 원본 신호에 랜덤한 변동이 섞여 잡음이 추가된 형태를 나타냅니다.
# 필터링된 신호 (Filtered Signal):
# Low-pass 필터링을 적용하여 고주파 성분을 제거하고 저주파 성분을 보존합니다.
# 필터의 컷오프 주파수(cutoff frequency)를 200 Hz로 설정했으며, 이 값보다 낮은 주파수 성분은 보존됩니다.
# 결과적으로, 고주파 성분(잡음)이 감소되고, 원본 신호의 주파수보다 낮은 주파수 성분만 남아있는 형태를 나타냅니다.
# 위의 과정을 통해 초음파 신호를 생성하고, 잡음을 추가하며, 그 후 필터링을 통해 신호를 처리한 결과를 확인할 수 있습니다.
# 이렇게 처리된 신호는 원본 신호의 특징을 유지하면서 노이즈를 줄인 상태로 분석이나 모니터링에 활용할 수 있습니다.
# 실제 응용에서는 필터링 방법과 파라미터를 조정하여 원하는 결과를 얻을 수 있습니다.
