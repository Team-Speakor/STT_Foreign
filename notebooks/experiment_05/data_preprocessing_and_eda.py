import json
import pandas as pd
import numpy as np
import torch
import evaluate
from datasets import Dataset, Audio
from transformers import (Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments, TrainerCallback)
from dataclasses import dataclass
from typing import Dict, List, Union
import re

# 경로 설정
train_audio_dir = "/home/ace3_yongjae/speechRecog/train/training"
train_json_dir = "/home/ace3_yongjae/speechRecog/train/labeling"
valid_audio_dir = "/home/ace3_yongjae/speechRecog/valid/validation"
valid_json_dir = "/home/ace3_yongjae/speechRecog/valid/labeling"

# JSON 로드 함수
def load_json_files(directory):
    return [json.load(open(os.path.join(directory, f), encoding='utf-8'))
            for f in os.listdir(directory) if f.endswith('.json')]

train_json = load_json_files(train_json_dir)
valid_json = load_json_files(valid_json_dir)

# DataFrame 생성 (라벨 누락 방지)
def create_dataframe(json_data, audio_dir):
    data = []
    for item in json_data:
        file_name = item.get('fileName')
        transcription = item.get('transcription', {})
        answer_text = transcription.get('AnswerLabelText') or transcription.get('ReadingLabelText', '')
        audio_path = os.path.join(audio_dir, file_name)
        # 라벨이 비어있거나 None인 경우 제외
        if os.path.exists(audio_path) and answer_text and isinstance(answer_text, str) and answer_text.strip():
            data.append({'file_path': audio_path, 'text': answer_text})
    return pd.DataFrame(data)

train_df = create_dataframe(train_json, train_audio_dir)
valid_df = create_dataframe(valid_json, valid_audio_dir)

# 텍스트 정규화
def prepare_korean_text(text):
    text = re.sub(r'[^\uAC00-\uD7A3\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

train_df['normalized_text'] = train_df['text'].apply(prepare_korean_text)
valid_df['normalized_text'] = valid_df['text'].apply(prepare_korean_text)