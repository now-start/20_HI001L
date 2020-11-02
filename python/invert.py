# 프로그램 상에서 음원을 합치는 코드

# 위상 반전, 파장 결합(Merge), 소리 재생 하는 소스코드

from pydub import AudioSegment
from pydub.playback import play

# 왼쪽에선 정상위상 오른쪽에선 반대위상 동시에 재생 시키기 이코드가 서버로 이동

# 기존 wav 파일 이름 지정
originalSoundFileName = "test.wav"

# 지정한 wav 파일 load
originalSound = AudioSegment.from_file(originalSoundFileName, format="wav")

# 기존 wav 파일 역위상 파장 생성
reversedSound = originalSound.invert_phase()

# 역위상 파장 wav파일로 저장
reversedSound.export("test_reverse.wav", format="wav")


# 두개의 소리(wav파일)를 결합(Merge)
combinedSound = originalSound.overlay(reversedSound)
# 결합된 소리를 저장
combinedSound.export("combined.wav", format="wav")

# Play audio file :
# should play nothing since two files with inverse phase cancel each other

# 기존 wav 파일 재생
play(originalSound)
# 위상 반전된 wav 파일 재생
play(reversedSound)
# 결합된 wav 파일 재생
play(combinedSound)
