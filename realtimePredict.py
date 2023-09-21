import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from datetime import datetime
from torch.autograd import Variable
from TransformerModel import TransformerModel  # 앞서 정의한 모델을 불러옵니다.

# 학습된 모델 불러오기
model = TransformerModel()
model.load_state_dict(torch.load('transformer_model.pth'))
model.eval()

# 예측할 데이터 가져오기 (실시간 데이터 수집 대신 예시 데이터 사용)
def get_realtime_data():
    # 여기에서 실제 데이터를 수집하거나 API를 통해 데이터를 가져올 수 있습니다.
    # 예시 데이터를 사용하므로 데이터를 직접 생성합니다.
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pm10 = random.random()*100  # 예시 미세먼지 데이터
    odor = round(random.random(), 2)  # 예시 악취 데이터

    return timestamp, pm10, odor

# 모델 입력을 위한 데이터 전처리
def prepare_input_data(pm10, odor):
    input_data = np.array([[pm10, odor]])  # 입력 데이터를 모델의 형식에 맞게 변환
    input_data = input_data / np.max(input_data)  # 데이터 정규화
    return torch.FloatTensor(input_data).unsqueeze(0)  # 모델 입력을 위해 텐서로 변환

# 예측 함수 정의
def predict_realtime_data():
    timestamp, pm10, odor = get_realtime_data()  # 실시간 데이터 가져오기
    input_data = prepare_input_data(pm10, odor)  # 데이터 전처리 및 텐서로 변환

    with torch.no_grad():
        predicted_value = model(input_data).item()  # 모델로 예측 수행

    return timestamp, predicted_value

# 실시간 예측 및 출력
while True:
    timestamp, predicted_value = predict_realtime_data()
    print(f'Timestamp: {timestamp}, Predicted Value: {predicted_value}')

    # 일정 시간 간격마다 예측 수행 (예: 5분마다)
    import time
    time.sleep(3)
