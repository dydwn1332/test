import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd

# 데이터 로드 (엑셀 파일)
data = pd.read_excel('Chungbuk_Fixed_Sensor_Data_Standard.xlsx')

# 미세먼지와 악취 데이터 추출 및 정규화
pm10_data = pd.to_numeric(data['미세먼지(PM10,㎍/㎥)'], errors='coerce')
pm10_data = pm10_data.fillna(pm10_data.mean())
pm10_data = pm10_data.values / np.max(pm10_data.values)

odor_data = pd.to_numeric(data['복합악취(OU)'], errors='coerce')
odor_data = odor_data.fillna(odor_data.mean())
odor_data = odor_data.values / np.max(odor_data.values)

# 합치기 및 reshape: (seq_length, 2) 형태로 변환
sequence_data = np.stack((pm10_data, odor_data), axis=-1).reshape(-1, 2)


class TransformerModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(TransformerModel, self).__init__()

        self.attn_transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=1,
                dim_feedforward=512),
            num_layers=3)

        self.linear_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Transpose the sequence to match the required input shape of Transformer.
        x = x.transpose(0, 1)

        x = self.attn_transformer_encoder(x)

        # Only use the last output of the sequence for prediction.
        x = x[-1]

        x = self.linear_layer(x)

        return x


def create_inout_sequences(input_sequence, tw):
    inout_seq = []
    L = len(input_sequence)

    for i in range(L - tw):
        train_seq = input_sequence[i:i + tw]
        train_label = input_sequence[i + tw - 1][1]
        inout_seq.append((train_seq, train_label))

    return inout_seq


class TransformerTrainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self, inout_seq, epochs):
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = []
            
            for i_batched, data in enumerate(inout_seq):
                seq, label = data
                seq, label = torch.FloatTensor(seq).unsqueeze(0), torch.FloatTensor([label]).unsqueeze(-1)
                self.optimizer.zero_grad()
                y_pred = self.model(seq)
                single_loss = self.criterion(y_pred, label)
                single_loss.backward()
                self.optimizer.step()
                epoch_loss.append(single_loss.item())

            print(f"Epoch {epoch} Loss {np.mean(epoch_loss)}")

    def predict(self, seq):
        self.model.eval()
        with torch.no_grad():
            seq = torch.FloatTensor(seq).unsqueeze(0)
            y_pred = self.model(seq)
        return y_pred.item()


seq_len = 20
inout_seq = create_inout_sequences(sequence_data, seq_len)

model = TransformerModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.001)

trainer = TransformerTrainer(model, criterion, optimizer)

epochs = 1
trainer.fit(inout_seq, epochs)

# 모델을 저장할 수 있으며 필요한 경우 저장 코드 추가
# torch.save(model.state_dict(), 'transformer_model.pth')

# 예측을 수행할 시퀀스를 만들고 predict 메서드를 사용하여 예측
# 예측 결과는 trainer.predict(seq)를 호출하여 얻을 수 있음
