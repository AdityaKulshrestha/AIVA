import tkinter as tk
from tkinter import ttk, scrolledtext
import pyaudio
import wave
import requests
import threading
import os
from datetime import datetime
from dotenv import load_dotenv
from agents.main import llm_anthropic
from agents.execute import natural_language_research

load_dotenv()


class AudioRecorderApp:
    def __init__(self, master):
        self.master = master
        master.title("Audio Recorder and Transcriber")
        master.geometry("400x300")

        self.is_recording = False
        self.frames = []

        # Audio recording configuration
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100

        self.create_widgets()

    def create_widgets(self):
        # Record button
        self.record_button = ttk.Button(self.master, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(pady=10)

        # Transcribe button
        self.transcribe_button = ttk.Button(self.master, text="Get Response", command=self.transcribe, state=tk.DISABLED)
        self.transcribe_button.pack(pady=10)

        # Status label
        self.status_label = ttk.Label(self.master, text="")
        self.status_label.pack(pady=5)

        # Transcription display
        self.transcription_display = scrolledtext.ScrolledText(self.master, wrap=tk.WORD, width=40, height=10)
        self.transcription_display.pack(pady=10, padx=10, expand=True, fill=tk.BOTH)

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.record_button.config(text="Stop Recording")
        self.status_label.config(text="Recording...")
        self.frames = []

        threading.Thread(target=self._record).start()

    def _record(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)

        while self.is_recording:
            data = stream.read(self.CHUNK)
            self.frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def stop_recording(self):
        self.is_recording = False
        self.record_button.config(text="Start Recording")
        self.status_label.config(text="Recording stopped")
        self.transcribe_button.config(state=tk.NORMAL)

    def save_audio(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tmp/audio_{timestamp}.wav"

        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        return filename

    def send_for_transcription(self, filename):
        # Replace with your actual endpoint URL
        endpoint_url = "https://api.sarvam.ai/speech-to-text"

        headers = {
            "api-subscription-key": os.environ['SARVAM_API_KEY'],
        }

        with open(filename, 'rb') as audio_file:
            files = {
                'file': (filename, audio_file, 'audio/wav')  # Provide the file name, file object, and MIME type
            }
            data = {
                # 'prompt': 'Translate the input language into English',  # Include the prompt if provided
                # 'model': 'saaras:v1'
                'language_code': 'hi-IN',
                'model': 'saarika:v1'
            }
            response = requests.request("POST", endpoint_url, data=data, files=files, headers=headers)

        if response.status_code == 200:
            return response.text
        else:
            return f"Error: {response.status_code} - {response.text}"

    def transcribe(self):
        self.status_label.config(text="Saving audio...")
        filename = self.save_audio()

        self.status_label.config(text="Transcribing...")
        transcription = self.send_for_transcription(filename)
        translated_text = llm_anthropic.invoke(
            f"""Strictly convert this input sentence into English. Do not give anything excess in response apart from the translated text
            Consider the following example:
            Input: आज के समाचार बताओ Output: Tell me the news for today
            Input: रिपोर्ट पीडीएफ फ़ाइल ढूंढो Output: Search for report.pdf
            Input: डेस्कटॉप पर रिपोर्ट डॉट पीडीएफ को प्रिंट करो Output: Print the report.pdf on desktop
            Input: {transcription} Output: """, max_tokens=50)
        response = natural_language_research(translated_text.content)

        self.transcription_display.delete('1.0', tk.END)
        self.transcription_display.insert(tk.END,
                                          f"Original {transcription} Translated: {translated_text.content} Response: {response}")

        self.status_label.config(text="Transcription complete")
        self.transcribe_button.config(state=tk.DISABLED)

        # Optionally, remove the audio file after transcription
        # os.remove(filename)


def main():
    root = tk.Tk()
    app = AudioRecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
