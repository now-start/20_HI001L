import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.models import load_model
import datetime
from scipy.io import wavfile

# wav 파일을 읽은후 pcm(audio_samples) 값을 csv로 저장
sample_rate, audio_samples = wavfile.read('Beep.wav', 'rb')

# 파형 출력 (그래프)
sample_rate, audio_samples = wavfile.read("Beep.wav", 'rb')

#! 수정 (오디오 정보, 파형 출력)
# 오디오 정보 출력.
duration = len(audio_samples)/sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(audio_samples)}')
# 파형 그래프 출력
plt.plot(audio_samples)
plt.show()

#! 수정 (변수명)
pd.DataFrame(audio_samples).to_csv("Beep.csv")
data = pd.read_csv('Beep.csv')
print(data['0'].describe())
train_data = data['0'].values
print(train_data)

#! 수정 (50개 -> 4/5)
# # 4/5의 데이터를 보고 다음을 예측
seq_len = (len(audio_samples) // 5) * 4
print(seq_len)
sequence_length = seq_len + 1

# 정확한 예측를 위해 값들을 정규화
result = []
for index in range(len(train_data) - sequence_length):
    result.append(train_data[index: index + sequence_length])


# normalized_data = []
# for i in range(len(result)):
#     # normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
#     print(i)
#     #! 수정 (minmax정규화) -> 기존 야매정규화 했을때 분모가 0(조용한 상태일 때)이 되어 ZeroDivisionError 발생가능
#     normalized_window = [
#         ((float(p) - float(min(result[i]))) / (float(max(result[i])) - float(min(result[i])))) for p in result[i]]
#     normalized_data.append(normalized_window)

# print('끝')
# result = np.array(normalized_data)

# 트레이닝할 값과 테스트 값을 나눠줌
row = int(round(np.array(result).shape[0] * 0.9))
train = result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

x_train.shape, x_test.shape

# 모델 생성
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')
model.summary()

# 트레이닝 값으로 학습
model.fit(x_train, y_train, validation_data=(
    x_test, y_test), batch_size=10, epochs=10)

# 모델 저장
model.save('weight.h5')
# 모델 로드
# model = load_model('weight.h5')

# 테스트 값으로 예측
pred = model.predict(x_test)
print(pred)

# 그래프
fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()
