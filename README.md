# 인공지능스피커를 이용한 생활소음 감소

## Setup
- Dependancy Installation - ``pip install -r requirements.txt``
- Running - ``python ___.py``

## Usage
- Run server.py or server.exe specifying the port you want to bind to.
- If you intend to use this program across the internet, ensure you have port forwarding that is forwarding the port the server is running on to the server's machine local IP (the IP displayed on the server program) and the correct port.
- Clients can connect across the internet by entering your public IP (as long as you have port forwarding to your machine) and the port the machine is running on or in the same network by entering the IP displayed on the server.
- If the client displays ``"Connected to Server"``, you can now communicate with others in the same server by speaking into a connected microphone.

## Requirements
- Python 3
- PyAudio
- pydub
- tensorflow

## Contributing


## License

