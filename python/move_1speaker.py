# 양쪽에서 들리는 소음을 한쪽 (왼쪽)으로 옮기는 소스 코드

# 음성(소음) 녹음, 재생 하는 패키지(wav파일)
import pyaudio
import wave

# 위상 반전, 파장 결합(Merge), 소리 재생 하는 패키지
from pydub import AudioSegment
from pydub.playback import play

from scipy.io import wavfile

import matplotlib.pyplot as plt

ORIGINAL_FILENAME = 'y_val.wav'
OUTPUT_FILENAME = "y_val_left.wav"
# original wav 파일 load
original_sound = AudioSegment.from_file(ORIGINAL_FILENAME, format="wav")

# 정 위상을 왼쪽에서 재생(스테레오)(pan 100 % left)
pannedLeft = original_sound.pan(-1)  # -1은 100% 왼쪽으로 이동 시킨다는 의미
# play(pannedLeft)

pannedLeft.export(OUTPUT_FILENAME, format="wav")
