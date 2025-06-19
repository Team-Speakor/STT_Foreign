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

# DataFrame 생성
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

train_df = create_dataframe(train_json, train_audio_dir)
valid_df = create_dataframe(valid_json, valid_audio_dir)

# 텍스트 정규화
def prepare_korean_text(text):
    text = re.sub(r'[^\uAC00-\uD7A3\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

train_df['normalized_text'] = train_df['text'].apply(prepare_korean_text)
valid_df['normalized_text'] = valid_df['text'].apply(prepare_korean_text)