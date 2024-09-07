import pyaudio
import wave
import requests
import os
from datetime import datetime

# Audio recording configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5


def record_audio():
    """Record audio from the microphone and return the frames."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    return frames


def save_audio(frames, filename):
    """Save the recorded audio frames to a WAV file."""
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio saved as {filename}")


def send_for_transcription(filename, endpoint_url):
    """Send the audio file to the specified endpoint for transcription."""
    with open(filename, 'rb') as audio_file:
        files = {'audio': audio_file}
        response = requests.post(endpoint_url, files=files)

    if response.status_code == 200:
        return response.json()['transcription']
    else:
        return f"Error: {response.status_code} - {response.text}"


def main():
    # Generate a unique filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"audio_{timestamp}.wav"

    # Record audio
    frames = record_audio()

    # Save audio to file
    save_audio(frames, filename)

    # Send for transcription (replace with your actual endpoint URL)
    endpoint_url = "https://your-transcription-api-endpoint.com/transcribe"
    transcription = send_for_transcription(filename, endpoint_url)

    print("Transcription:", transcription)

    # Optionally, remove the audio file after transcription
    # os.remove(filename)


# if __name__ == "__main__":
    # main()