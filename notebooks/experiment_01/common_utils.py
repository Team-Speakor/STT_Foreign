import os
import json
import re

def load_json_files(directory):
    json_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_data.append(data)
    return json_data

def create_dataframe(json_data, audio_dir):
    import pandas as pd
    data = []
    for item in json_data:
        file_name = item.get('fileName')
        answer_text = item.get('transcription', {}).get('AnswerLabelText', '')
        audio_path = os.path.join(audio_dir, file_name)
        if os.path.exists(audio_path) and answer_text:
            data.append({
                'file_path': audio_path,
                'text': answer_text
            })
    return pd.DataFrame(data)

def prepare_korean_text(text):
    text = re.sub(r'[^\uAC00-\uD7A3\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text