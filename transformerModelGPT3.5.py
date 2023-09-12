import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성 (간단한 시계열 데이터 예시)
seq_length = 100
time_steps = np.linspace(0, 10, seq_length)
sin_wave = np.sin(time_steps) + np.random.normal(scale=0.1, size=seq_length)

# 데이터 전처리
X = torch.tensor(sin_wave[:-1], dtype=torch.float32).view(-1, 1, 1)  # 입력 시계열 데이터
y = torch.tensor(sin_wave[1:], dtype=torch.float32).view(-1, 1, 1)   # 타겟 시계열 데이터

# 트랜스포머 모델 정의
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers):
        super(TransformerTimeSeries, self).__init__()
        self.d_model ㅁ= input_dim * num_heads  # d_model을 num_heads의 배수로 설정
        self.transformer = nn.Transformer(d_model=self.d_model, nhead=num_heads, num_encoder_layers=num_layers)
        self.fc = nn.Linear(self.d_model, 1)

    def forward(self, src):
        tgt = src  # src와 tgt의 차원을 동일하게 설정
        out = self.transformer(src, tgt)
        out = self.fc(out)
        return out

# 모델 초기화
input_dim = 1      # 입력 데이터의 차원
num_heads = 4      # 어텐션 헤드 수
num_layers = 2     # 트랜스포머 층 수
model = TransformerTimeSeries(input_dim, num_heads, num_layers)

# 손실 함수 정의 (평균 제곱 오차)
criterion = nn.MSELoss()

# 옵티마이저 정의 (Adam 사용)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X)  # src에 데이터(X)를 전달
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'에폭 [{epoch+1}/{num_epochs}], 손실: {loss.item():.4f}')

# 예측 결과 시각화
model.eval()
with torch.no_grad():
    predicted = model(X)  # 예측을 위해 src에 데이터(X)를 전달

plt.figure(figsize=(12, 6))
plt.plot(time_steps[1:], y.view(-1).numpy(), label='타겟 시계열 데이터', color='blue')
plt.plot(time_steps[1:], predicted.view(-1).numpy(), label='예측', color='green')
plt.xlabel('타임 스텝')
plt.ylabel('값')
plt.legend()
plt.title('시계열 데이터와 트랜스포머 예측 결과')
plt.show()
