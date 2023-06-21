import os
import torch
from transformers import pipeline

MODEL_NAME = "biodatlab/whisper-th-medium-combined"
lang = "th"

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

# Upload csv data
with st.sidebar.header('Upload your MP3'):
    uploaded_file = st.sidebar.file_uploader("Upload your input mp3 file", type=["mp3"])

# Pandas Profiling Report
if uploaded_file is not None:
    transcriptions = pipe(
        "uploaded_file",
        batch_size=16,
        return_timestamps=False,
        generate_kwargs={"language": "<|th|>", "task": "transcribe"}
    )["text"]
    print(transcriptions)
    
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')

