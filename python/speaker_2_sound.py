# speaker_2_sound.py
# 한 스피커로 녹음해서 정위상, 역위상 wav를 생성한 다음 정위상은 왼쪽, 역위상은 오른쪽 스피커에서 재생시키는 소스코드
# (정위상, 역위상 파일을 하나의 스테레오 wav로 만듦)

# 음성(소음) 녹음, 재생 하는 패키지(wav파일)
import pyaudio
import wave

# 위상 반전, 파장 결합(Merge), 소리 재생 하는 패키지
from pydub import AudioSegment
from pydub.playback import play

from scipy.io import wavfile

import matplotlib.pyplot as plt

CHUNK = 1024
FORMAT = pyaudio.paInt16  # Portaudio Sample Format 설정
CHANNELS = 1  # 채널
RATE = 44100
RECORD_SECONDS = 5  # 녹음 시간(초)

thread = None

# 녹음한 wav 파일 이름 지정
WAVE_OUTPUT_FILENAME = "originalAudio.wav"

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

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


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
