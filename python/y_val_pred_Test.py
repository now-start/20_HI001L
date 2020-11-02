# 기존 y_val 은 그대로, pred를 역위상으로 만든 다음 스피커 각각에서 동시재생하는 소스 코드

# 음성(소음) 녹음, 재생 하는 패키지(wav파일)
import pyaudio
import wave

# 위상 반전, 파장 결합(Merge), 소리 재생 하는 패키지
from pydub import AudioSegment
from pydub.playback import play

from scipy.io import wavfile

import matplotlib.pyplot as plt

Y_VAL_FILENAME = 'y_val.wav'
PRED_FILENAME = 'pred.wav'


# y_val wav 파일 load
y_val_sound = AudioSegment.from_file(Y_VAL_FILENAME, format="wav")

# pred wav 파일 load
pred_sound = AudioSegment.from_file(PRED_FILENAME, format="wav")

# 기존 pred wav 파일 역위상 파장 생성
reversed_pred_sound = pred_sound.invert_phase()

# 역위상 파장 wav파일로 저장 (생략 가능)
reversed_pred_sound.export("reversed_pred.wav", format="wav")

# y_val 재생
# play(y_val_sound)
# pred 재생
# play(reversed_pred_sound)

# 정 위상을 왼쪽에서 재생 (스테레오) (pan 100% left)
# pannedLeft = originalSound.pan(-1)  # -1은 100% 왼쪽으로 이동 시킨다는 의미
# play(pannedLeft)

# 정 위상을 왼쪽에서 재생 (스테레오) (pan 100% right)
# pannedRight = reversedSound.pan(1)  # +1은 100% 오른쪽으로 이동 시킨다는 의미
# play(pannedRight)

# 스테레오 두 파일을 왼쪽에서 들리는 모노, 오른쪽에서만 들리는 모노로 바꾼다음 합쳐서 하나의 스테레오 파일로 만듦
stereo_sound = AudioSegment.from_mono_audiosegments(
    y_val_sound, reversed_pred_sound)
play(stereo_sound)
stereo_sound.export("stereo_sound_AI.wav", format="wav")


# 파형 출력 (그래프)
sample_rate, audio_samples = wavfile.read("stereo_sound_AI.wav", 'rb')

# Show some basic information about the audio.
duration = len(audio_samples)/sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(audio_samples)}')

plt.plot(audio_samples)
plt.show()
