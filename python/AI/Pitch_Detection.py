import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa import display as librosadisplay

import logging
import math
import statistics
import sys

from IPython.display import Audio, Javascript
from scipy.io import wavfile

from base64 import b64decode

import music21
from pydub import AudioSegment

# 음성(소음) 녹음, 재생 하는 패키지(wav파일)
import pyaudio
import wave

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

print("tensorflow: %s" % tf.__version__)
# print("librosa: %s" % librosa.__version__)


CHUNK = 1024
FORMAT = pyaudio.paInt16  # Portaudio Sample Format 설정
CHANNELS = 1  # 채널
RATE = 44100
RECORD_SECONDS = 10  # 녹음 시간(초)

thread = None

# 녹음한 wav 파일 이름 지정
uploaded_file_name = "test.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,  # input 스트림 명시
                frames_per_buffer=CHUNK)

print("Start to record the audio.")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording is finished.")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(uploaded_file_name, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


# Function that converts the user-created audio to the format that the model
# expects: bitrate 16kHz and only one channel (mono).

EXPECTED_SAMPLE_RATE = 16000


def convert_audio_for_model(user_file, output_file='converted_audio_file.wav'):
    audio = AudioSegment.from_file(user_file, format="wav")
    audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE).set_channels(1)
    audio.export(output_file, format="wav")
    return output_file


# Converting to the expected format for the model
# in all the input 4 input method before, the uploaded file name is at
# the variable uploaded_file_name
converted_audio_file = convert_audio_for_model(uploaded_file_name)

# # Loading audio samples from the wav file:
sample_rate, audio_samples = wavfile.read(converted_audio_file, 'rb')

# # Show some basic information about the audio.
duration = len(audio_samples)/sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(audio_samples)}')

# # Let's listen to the wav file.
print(Audio(audio_samples, rate=sample_rate))

# We can visualize the audio as a waveform.
_ = plt.plot(audio_samples)


MAX_ABS_INT16 = 32768.0


def plot_stft(x, sample_rate, show_black_and_white=False):
    x_stft = np.abs(librosa.stft(x, n_fft=2048))
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    x_stft_db = librosa.amplitude_to_db(x_stft, ref=np.max)
    if(show_black_and_white):
        librosadisplay.specshow(data=x_stft_db, y_axis='log',
                                sr=sample_rate, cmap='gray_r')
    else:
        librosadisplay.specshow(data=x_stft_db, y_axis='log', sr=sample_rate)

    plt.colorbar(format='%+2.0f dB')


plot_stft(audio_samples / MAX_ABS_INT16, sample_rate=EXPECTED_SAMPLE_RATE)
plt.show()

# 정규화
audio_samples = audio_samples / float(MAX_ABS_INT16)

# 모델 로드
model = hub.load("https://tfhub.dev/google/spice/2")


# We now feed the audio to the SPICE tf.hub model to obtain pitch and uncertainty outputs as tensors.
model_output = model.signatures["serving_default"](
    tf.constant(audio_samples, tf.float32))

pitch_outputs = model_output["pitch"]
uncertainty_outputs = model_output["uncertainty"]

# 'Uncertainty' basically means the inverse of confidence.
confidence_outputs = 1.0 - uncertainty_outputs

fig, ax = plt.subplots()
fig.set_size_inches(20, 10)
plt.plot(pitch_outputs, label='pitch')
plt.plot(confidence_outputs, label='confidence')
plt.legend(loc="lower right")
plt.show()
