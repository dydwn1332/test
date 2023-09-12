import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

# 데이터 생성 함수 (sine wave)
def generate_data(seq_length):
    freq = 5 
    noise = 0.05

    x = np.linspace(0, 4*np.pi*freq, seq_length)
    y = np.sin(x) + noise * np.random.randn(seq_length)

    split_boundary = int(seq_length * 0.8)

    train_sequence = y[:split_boundary]
    test_sequence = y[split_boundary:]

    return train_sequence, test_sequence

# 트랜스포머 모델 정의
class TransformerModel(nn.Module):

   def __init__(self, input_dim=1 ,output_dim=1 ,seq_len=100):
       super(TransformerModel,self).__init__()

       self.seq_len=seq_len
      
       self.attn_transformer_encoder=nn.TransformerEncoder(
           nn.TransformerEncoderLayer(d_model=input_dim,nhead=1,dim_feedforward=512),
           num_layers=3)
      
       self.linear_layer = nn.Linear(input_dim, output_dim)

   def forward(self,x):
       x = self.attn_transformer_encoder(x)
       
       # Only use the last output of the sequence
       x = x[-1]
       
       x = self.linear_layer(x)
      
       return x

   
# 시퀀스 데이터를 입력과 타깃으로 분할하는 함수 
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
        
    return inout_seq

# 학습 및 테스트 데이터 생성
train_sequence, test_sequence = generate_data(10000)

train_sequence=torch.FloatTensor(train_sequence).view(-1)
test_sequence=torch.FloatTensor(test_sequence).view(-1)

# 시퀀스 길이 설정 및 입력-타깃 분할 실행
seq_len   = 20 

train_inout_seq=create_inout_sequences(train_sequence, seq_len)
test_inout_seq=create_inout_sequences(test_sequence, seq_len)


model     = TransformerModel()
criterion=torch.nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=.001) 

epochs     = 10 

for i in range(epochs):
  
   model.train()

   epoch_loss     	=[]
   
   loop=tqdm(enumerate(train_inout_seq),total=len(train_inout_seq),leave=False) 
  
   for i,data in loop:
       
      seq,label=data
      
      # Add the necessary dimensions to the sequence and convert label to tensor form.
      seq,label=torch.unsqueeze(torch.unsqueeze(seq,-1),0),label.view(-1)

      optimizer.zero_grad()
      
      y_pred=model(seq).squeeze()

      single_loss=criterion(y_pred,label) 
      single_loss.backward()
      
      optimizer.step()
      
      epoch_loss.append(single_loss.item())
   
   print(f"Epoch {i} Loss {np.mean(epoch_loss)}")
