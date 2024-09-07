import wave
import requests
import os
from datetime import datetime
from voice.main import record_audio, save_audio, send_for_transcription


def main():
    # Generate a unique filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('tmp', exist_ok=True)
    filename = f"tmp/audio_{timestamp}.wav"

    # Record audio
    frames = record_audio()

    # Save audio to file
    save_audio(frames, filename)

    # Send for transcription (replace with your actual endpoint URL)
    endpoint_url = "https://your-transcription-api-endpoint.com/transcribe"
    transcription = send_for_transcription(filename, endpoint_url)

    print("Transcription:", transcription)

    # Optionally, remove the audio file after transcription
    os.remove(filename)

if __name__ == "__main__":
    main()
