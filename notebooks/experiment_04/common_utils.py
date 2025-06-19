import os
import json
import re
import pandas as pd

def load_json_files(directory):
    return [json.load(open(os.path.join(directory, f), encoding='utf-8'))
            for f in os.listdir(directory) if f.endswith('.json')]

def create_dataframe(json_data, audio_dir):
    data = []
    for item in json_data:
        file_name = item.get('fileName')
        transcription = item.get('transcription', {})
        answer_text = transcription.get('AnswerLabelText') or transcription.get('ReadingLabelText', '')
        audio_path = os.path.join(audio_dir, file_name)
        if os.path.exists(audio_path) and answer_text:
            data.append({'file_path': audio_path, 'text': answer_text})
    return pd.DataFrame(data)

def prepare_korean_text(text):
    text = re.sub(r'[^\uAC00-\uD7A3\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()