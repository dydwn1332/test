import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 데이터 생성 (간단한 시계열 데이터 예시)
seq_length = 100
time_steps = np.linspace(0, 10, seq_length)
sin_wave = np.sin(time_steps) + np.random.normal(scale=0.1, size=seq_length)

# 데이터 전처리
X = torch.tensor(sin_wave[:-1], dtype=torch.float32).view(1, -1, 1)  # 입력 시계열 데이터
y = torch.tensor(sin_wave[1:], dtype=torch.float32).view(1, -1, 1)   # 타겟 시계열 데이터

# 트랜스포머 모델 정의
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=2, num_encoder_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        out = self.transformer(src, tgt)
        out = self.fc(out)
        return out

# 모델 초기화
input_dim = 1      # 입력 데이터의 차원
hidden_dim = 32    # 히든 차원
num_layers = 1     # 트랜스포머 층 수
model = SimpleTransformer(input_dim, hidden_dim, num_layers)

# 손실 함수 정의 (평균 제곱 오차)
criterion = nn.MSELoss()

# 옵티마이저 정의 (Adam 사용)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 1000
losses = []

for epoch in tqdm(range(num_epochs), desc="Training"):
    optimizer.zero_grad()
    outputs = model(X, X)  # src와 tgt에 동일한 데이터(X)를 전달
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# 예측 결과 시각화
model.eval()
with torch.no_grad():
    predicted = model(X, X)  # 예측을 위해 src와 tgt에 동일한 데이터(X)를 전달

plt.figure(figsize=(12, 6))
plt.plot(time_steps[1:], y.view(-1).numpy(), label='타겟 시계열 데이터', color='blue')
plt.plot(time_steps[1:], predicted.view(-1).numpy(), label='예측', color='green')
plt.xlabel('타임 스텝')
plt.ylabel('값')
plt.legend()
plt.title('시계열 데이터와 트랜스포머 예측 결과')
plt.show()

# 손실 그래프 시각화
plt.figure(figsize=(12, 6))
plt.plot(range(num_epochs), losses, label='손실')
plt.xlabel('에폭')
plt.ylabel('손실')
plt.legend()
plt.title('학습 중 손실 그래프')
plt.show()
