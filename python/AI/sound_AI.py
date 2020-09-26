import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.models import load_model
import datetime
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment
import tensorflow as tf

uploaded_file_name = 'test.wav'
EXPECTED_SAMPLE_RATE = 44100


def convert_audio_for_model(user_file, output_file='test_re.wav'):
    audio = AudioSegment.from_file(user_file, format="wav")
    audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE).set_channels(1)
    audio.export(output_file, format="wav")
    return output_file


converted_audio_file = convert_audio_for_model(uploaded_file_name)

# # Loading audio samples from the wav file:
sample_rate, audio_samples = wavfile.read(converted_audio_file, 'rb')

# 파형 출력 (그래프)

#! 수정 (오디오 정보, 파형 출력)
# 오디오 정보 출력.
duration = len(audio_samples)/sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(audio_samples)}')
# 파형 그래프 출력
plt.plot(audio_samples)
plt.xlim(0, 20000)
plt.show()

pd.DataFrame(audio_samples).to_csv("original.csv")
data = pd.read_csv('original.csv')
print(data['0'].describe())
train_data = data['0'].values
print(train_data)

#! 수정 (50개 -> 5초간)
# # 5초 간의 데이터를 보고 다음을 예측

seq_len = 50  # 예측을 위한 데이터 수
prediction = 1  # 다음 예측할 데이터 수
print("seq_len : ", seq_len)
sequence_length = seq_len + prediction


result = []
for index in range(len(train_data) - sequence_length):
    result.append(train_data[index: index + sequence_length])

# 정확한 예측를 위해 값들을 표준화
# standard_data = []
# for i in result:
#     # [((float(p) / float(i[0])) - 1) for p in i]
#     scaler = StandardScaler()
#     standard_window = [(scaler.fit_transform(
#         p.reshape(-1, 1)).reshape(1)) for p in i]
#     standard_data.append(standard_window)

# scaler = StandardScaler()
# normalized_data = scaler.fit_transform(np.array(result))

#     print(i)
#     #! 수정 (표준화) -> 기존 정규화 했을때 분모가 0(조용한 상태일 때)이 되어 ZeroDivisionError 발생

#     # normalized_window = [
#     #     ((float(p) - float(min(result[i]))) / (float(max(result[i])) - float(min(result[i])))) for p in result[i]]
#     normalized_window = [
#         ((float(p) - np.array(result[i]).mean()) / np.array(result[i].std())) for p in result[i]]
#     normalized_data.append(normalized_window)

# print('끝')
# result = normalized_data

result = np.array(result)
# result_df = pd.DataFrame(result)
# result_df.dropna(axis=0, inplace=True)
# result = result_df.values

# 트레이닝할 값과 테스트 값을 나눠줌
row = int(round(result.shape[0] * 0.8))
train = result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-prediction]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -prediction]
x_val = result[row:, :-prediction]
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
y_val = result[row:, -prediction]
y_val = y_val.astype(np.float32)
wavfile.write("y_val.wav", EXPECTED_SAMPLE_RATE, y_val.reshape(-1, 1))
print("y_val dtype", y_val.dtype)
print("y_val shape", y_val.shape)
print("y_val shape : ", y_val.reshape(-1, 1).shape)

# with tf.device("/gpu:0"):
#     #   x_train.shape, x_val.shape
#     # 모델 생성
#     model = Sequential()
#     model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))
#     model.add(LSTM(64, return_sequences=False))
#     model.add(Dense(1, activation='linear'))
#     model.compile(loss='mse', optimizer='rmsprop')
#     model.summary()

#     # 트레이닝 값으로 학습
#     hist = model.fit(x_train, y_train, validation_data=(
#         x_val, y_val), batch_size=10, epochs=2000)

#     # 모델 저장
#     model.save('weight.h5')

#     fig, loss_ax = plt.subplots()

#     acc_ax = loss_ax.twinx()

#     loss_ax.plot(hist.history['loss'], 'y', label='train loss')
#     loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

#     acc_ax.plot(hist.history['acc'], 'b', label='train acc')
#     acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

#     loss_ax.set_xlabel('epoch')
#     loss_ax.set_ylabel('loss')
#     acc_ax.set_ylabel('accuracy')

#     loss_ax.legend(loc='upper left')
#     acc_ax.legend(loc='lower left')

#     plt.show()

# 모델 로드
model = load_model('weight.h5')

# 테스트 값으로 예측
pred = model.predict(x_val)
print(pred)

# 그래프
fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_val, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.xlim(0, 10000)
plt.show()

# 예측값을 wav로 출력

# audio_samples = np.array(audio_samples)
# print(type(audio_samples))
# print(type(pred))
wavfile.write("pred.wav", EXPECTED_SAMPLE_RATE, pred)
print("pred dtype", pred.dtype)
print("pred shape", pred.shape)
print("pred shape : ", pred.shape)

print("end")
