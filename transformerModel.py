import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import pandas as pd

# 데이터 로드 (엑셀 파일)
data = pd.read_excel('Chungbuk_Fixed_Sensor_Data_Standard.xlsx')

# 미세먼지와 악취 데이터 추출 및 정규화
pm10_data = pd.to_numeric(data['미세먼지(PM10,㎍/㎥)'],
                          errors='coerce')  # 문자열을 NaN으로 변환
pm10_data = pm10_data.fillna(pm10_data.mean())  # NaN 값을 평균 값으로 채우기
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

    for i in range(L-tw):
        train_seq = input_sequence[i:i+tw]
        train_label = input_sequence[i+tw-1][1]
        inout_seq.append((train_seq, train_label))

    return inout_seq


seq_len = 20

inout_seq = create_inout_sequences(sequence_data, seq_len)


model = TransformerModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.001)


epochs = 1


for i in range(epochs):

    model.train()

    epoch_loss = []

    loop = tqdm(enumerate(inout_seq), total=len(inout_seq), leave=False)

    for i_batched, data in loop:
        seq, label = data

        seq, label = torch.FloatTensor(seq).unsqueeze(
            0), torch.FloatTensor([label]).unsqueeze(-1)

        optimizer.zero_grad()

        y_pred = model(seq)

        single_loss = criterion(y_pred, label)

        single_loss.backward()

        optimizer.step()

    epoch_loss.append(single_loss.item())

    print(f"Epoch {i} Loss {np.mean(epoch_loss)}")


torch.save(model.state_dict(), 'transformer_model_beforeEval.pth')
model.eval()

#모델 저장


# actual, predicted = [], []

# for i in range(len(inout_seq)):
#     seq, label = inout_seq[i]
#     seq, label = torch.FloatTensor(seq).unsqueeze(
#         0), torch.FloatTensor([label]).unsqueeze(-1)

#     with torch.no_grad():
#         y_pred = model(seq)

#     actual.append(label.item())
#     predicted.append(y_pred.item())

# plt.figure(figsize=(14, 5))
# plt.plot(actual, label='Actual')
# plt.plot(predicted, label='Predicted')
# plt.legend()
# plt.show()
