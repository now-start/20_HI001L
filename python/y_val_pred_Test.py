# 기존 y_val 은 정상위상, pred를 역위상으로 만든 다음 스피커 각각에서 동시재생

# 음성(소음) 녹음, 재생 하는 패키지(wav파일)
import pyaudio
import wave

# 위상 반전, 파장 결합(Merge), 소리 재생 하는 패키지
from pydub import AudioSegment
from pydub.playback import play

from scipy.io import wavfile

import matplotlib.pyplot as plt

Y_VAL_FILENAME

# 지정한 wav 파일 load
originalSound = AudioSegment.from_file(WAVE_OUTPUT_FILENAME, format="wav")

# 기존 wav 파일 역위상 파장 생성
reversedSound = originalSound.invert_phase()

# 역위상 파장 wav파일로 저장 (생략 가능)
reversedSound.export("reversedAudio.wav", format="wav")

# 정위상 재생
# play(originalSound)
# 역위상 재생
# play(reversedSound)

# 정 위상을 왼쪽에서 재생 (스테레오) (pan 100% left)
# pannedLeft = originalSound.pan(-1)  # -1은 100% 왼쪽으로 이동 시킨다는 의미
# play(pannedLeft)

# 정 위상을 왼쪽에서 재생 (스테레오) (pan 100% right)
# pannedRight = reversedSound.pan(1)  # +1은 100% 오른쪽으로 이동 시킨다는 의미
# play(pannedRight)

# 스테레오 두 파일을 왼쪽에서 들리는 모노, 오른쪽에서만 들리는 모노로 바꾼다음 합쳐서 하나의 스테레오 파일로 만듦
stereo_sound = AudioSegment.from_mono_audiosegments(
    originalSound, reversedSound)
play(stereo_sound)
stereo_sound.export("stereo_sound.wav", format="wav")


# 파형 출력 (그래프)
sample_rate, audio_samples = wavfile.read("stereo_sound.wav", 'rb')

# Show some basic information about the audio.
duration = len(audio_samples)/sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(audio_samples)}')

plt.plot(audio_samples)
plt.show()
