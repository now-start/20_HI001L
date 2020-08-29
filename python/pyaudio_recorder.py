# 음성(소음) 녹음하는 코드(wav파일)
import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16  # Portaudio Sample Format 설정
CHANNELS = 1  # 채널 수
RATE = 44100
RECORD_SECONDS = 5  # 녹음 시간(초)
WAVE_OUTPUT_FILENAME = "output.wav"

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
