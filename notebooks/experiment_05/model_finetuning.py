# 모델 로드
MODEL_ID = "kresnik/wav2vec2-large-xlsr-korean"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_ID,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id
)
model.freeze_feature_encoder()

# 데이터셋 전처리
def prepare_dataset(df):
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("file_path", Audio(sampling_rate=16000))
    return dataset

train_dataset = prepare_dataset(train_df)
valid_dataset = prepare_dataset(valid_df)

def prepare_dataset_for_model(batch):
    audio = batch["file_path"]
    array = audio["array"]
    if np.max(np.abs(array)) > 0:
        array = array / np.max(np.abs(array))
    batch["input_values"] = processor(array, sampling_rate=16000).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["normalized_text"]).input_ids
    return batch

train_dataset = train_dataset.map(prepare_dataset_for_model, remove_columns=train_dataset.column_names)
valid_dataset = valid_dataset.map(prepare_dataset_for_model, remove_columns=valid_dataset.column_names)

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")
        batch["labels"] = labels_batch["input_ids"]
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# 평가 메트릭
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    # -100을 pad_token_id로 변환
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    return {
        "wer": wer_metric.compute(predictions=pred_str, references=label_str),
        "cer": cer_metric.compute(predictions=pred_str, references=label_str)
    }

# 메모리 및 Loss 모니터링 콜백
class UnfreezeFeatureEncoderCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.epoch == 5:
            model = kwargs.get('model')
            if model is not None:
                for param in model.wav2vec2.feature_extractor.parameters():
                    param.requires_grad = True
                print("\nFeature encoder unfrozen!")

    def on_epoch_end(self, args, state, control, **kwargs):
        # Loss 직접 출력
        if hasattr(state, 'log_history') and state.log_history:
            last_log = state.log_history[-1]
            if 'loss' in last_log:
                print(f"Epoch {state.epoch} - Loss: {last_log['loss']:.6f}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} - Allocated: {torch.cuda.memory_allocated(i)/1e9:.2f} GB, Reserved: {torch.cuda.memory_reserved(i)/1e9:.2f} GB")
        torch.cuda.empty_cache()

# 학습 설정
training_args = TrainingArguments(
    output_dir="./wav2vec2-korean-asr",
    group_by_length=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    eval_strategy="epoch",
    num_train_epochs=4,
    fp16=False,
    eval_accumulation_steps=2,
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    learning_rate=3e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
    load_best_model_at_end=False,
    metric_for_best_model="wer",
    greater_is_better=False,
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=processor,
    callbacks=[UnfreezeFeatureEncoderCallback()]
)

#model.train()