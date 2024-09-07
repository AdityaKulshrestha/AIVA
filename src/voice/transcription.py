from openai import OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders
from dotenv import load_dotenv
import os

load_dotenv()

PORTKEY_API_KEY_OPENAI = os.getenv('PORTKEY_KEY_OPENAI')
VIRTUAL_KEY_OPENAI = os.getenv('PORTKEY_VIRTUAL_KEY_OPENAI')

audio_file= open("tmp/audio_20240908_005504.wav", "rb")

import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
    torch_dtype=torch.float16,
    device="cuda:0", # or mps for Mac devices
    model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
)

outputs = pipe(
    "tmp/audio_20240908_005504.wav",
    chunk_length_s=30,
    batch_size=24,
    return_timestamps=True,
)
print(outputs)

# client = OpenAI(
#     base_url=PORTKEY_GATEWAY_URL,
#     default_headers=createHeaders(
#         api_key=PORTKEY_API_KEY_OPENAI,
#         virtual_key=VIRTUAL_KEY_OPENAI
#     )
# )
#
#
# # Transcription
#
# transcription = client.audio.transcriptions.create(
#   model="whisper-1",
#   file=audio_file
# )
# print(transcription.text)
#
# # Translation
#
# translation = client.audio.translations.create(
#   model="whisper-1",
#   file=audio_file
# )
# print(translation.text)