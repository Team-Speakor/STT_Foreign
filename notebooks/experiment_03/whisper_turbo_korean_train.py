#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한국어 음성 인식 모델 훈련 스크립트 - Experiment 03: Whisper Turbo Korean
ghost613/whisper-large-v3-turbo-korean 모델 Fine-tuning
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import librosa
import re
from datetime import datetime
from datasets import Dataset, Audio
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    WhisperTokenizer,
    Trainer, 
    TrainingArguments, 
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import matplotlib.pyplot as plt

# 🔧 메모리 최적화 및 데드락 해결 설정
os.environ["NCCL_P2P_DISABLE"] = "1"  # 🚨 핵심: NCCL P2P 비활성화
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1 사용 (GPU 0은 현재 실험 중)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 토크나이저 병렬처리 비활성화
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP 스레드 수 제한

def clear_gpu_memory():
    """GPU 메모리 정리 함수"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        import gc
        gc.collect()

def print_gpu_memory_usage():
    """GPU 메모리 사용량 출력"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"💾 GPU 메모리: {allocated:.2f}GB 할당됨, {reserved:.2f}GB 예약됨, {total:.2f}GB 총 용량")

def main():
    print("=== 🚀 Experiment 03: Whisper Turbo Korean Fine-tuning ===")
    print("🔧 ghost613/whisper-large-v3-turbo-korean 모델 학습")
    
    # 초기 메모리 정리
    clear_gpu_memory()
    
    # GPU 설정 확인
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    print(f"GPU 개수: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"현재 GPU: {torch.cuda.current_device()}")
        print(f"GPU 이름: {torch.cuda.get_device_name()}")
        
        # bf16 지원 여부 확인
        try:
            bf16_supported = torch.cuda.is_bf16_supported()
            print(f"bf16 지원: {bf16_supported}")
        except:
            bf16_supported = False
            print(f"bf16 지원: 확인 불가 (False로 설정)")
    else:
        bf16_supported = False
    
    # ===== 1. 데이터 로딩 =====
    print("\n1. 데이터 로딩...")
    
    train_audio_dir = "../../data/train/audio"
    train_json_dir = "../../data/train/label"
    valid_audio_dir = "../../data/valid/audio" 
    valid_json_dir = "../../data/valid/label"
    
    def load_json_files(directory):
        json_data = []
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.json'):
                    with open(os.path.join(root, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        json_data.append(data)
        return json_data
    
    def create_dataframe(json_data, audio_base_dir):
        data = []
        for item in json_data:
            file_name = item.get('fileName')
            answer_text = item.get('transcription', {}).get('AnswerLabelText', '')
            
            if file_name and answer_text:
                audio_path = None
                for root, dirs, files in os.walk(audio_base_dir):
                    if file_name in files:
                        audio_path = os.path.join(root, file_name)
                        break
                
                if audio_path and os.path.exists(audio_path):
                    data.append({
                        'file_path': audio_path,
                        'text': answer_text
                    })
        
        return pd.DataFrame(data)
    
    train_json = load_json_files(train_json_dir)
    valid_json = load_json_files(valid_json_dir)
    
    train_df = create_dataframe(train_json, train_audio_dir)
    valid_df = create_dataframe(valid_json, valid_audio_dir)
    
    print(f"🔧 Whisper Turbo Korean: 전체 데이터 사용 - 학습: {len(train_df)}개, 검증: {len(valid_df)}개")
    
    # 텍스트 정규화 (Whisper용)
    def prepare_korean_text_for_whisper(text):
        # Whisper는 더 보수적인 정규화가 필요
        text = re.sub(r'[^\uAC00-\uD7A3\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    train_df['normalized_text'] = train_df['text'].apply(prepare_korean_text_for_whisper)
    valid_df['normalized_text'] = valid_df['text'].apply(prepare_korean_text_for_whisper)
    
    print(f"최종 데이터 크기 - 학습: {len(train_df)}, 검증: {len(valid_df)}")
    
    # ===== 2. Whisper 모델 로딩 =====
    print("\n2. Whisper Turbo Korean 모델 로딩...")
    
    MODEL_ID = "ghost613/whisper-large-v3-turbo-korean"
    
    # Processor와 Tokenizer 로딩
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID)
    
    # 4bit 양자화 설정 (PEFT 호환)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # 모델 로딩 (4bit 양자화 + PEFT)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # 🔧 Whisper 특화 설정
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # 학습 시 캐시 비활성화
    
    # 🔧 PEFT를 위한 모델 준비 (양자화된 모델)
    model = prepare_model_for_kbit_training(model)
    
    # 🔧 LoRA 설정 (Whisper용) - 올바른 task_type 사용
    lora_config = LoraConfig(
        r=32,  # LoRA rank - Whisper는 더 높은 rank 필요
        lora_alpha=64,  # LoRA alpha
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention layers
            "fc1", "fc2",  # feed forward layers
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM",  # Whisper는 sequence-to-sequence 모델
    )
    
    # 🔧 PEFT 모델 생성
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 학습 가능한 파라미터 수 출력
    
    print("✅ Whisper Turbo Korean 모델 로딩 완료")
    print(f"📊 모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 학습 가능 파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print_gpu_memory_usage()
    
    # ===== 3. 데이터셋 준비 =====
    print("\n3. Whisper 데이터셋 준비...")
    
    def prepare_dataset(df):
        dataset = Dataset.from_pandas(df)
        dataset = dataset.cast_column("file_path", Audio(sampling_rate=16000))
        return dataset
    
    train_dataset = prepare_dataset(train_df)
    valid_dataset = prepare_dataset(valid_df)
    
    def prepare_dataset_for_whisper(batch):
        # 오디오 처리
        audio = batch["file_path"]
        array = audio["array"]
        
        # 20초 제한 (OOM 방지를 위해 30초 → 20초)
        max_samples = 20 * 16000
        if len(array) > max_samples:
            array = array[:max_samples]
        
        # 정규화
        if np.max(np.abs(array)) > 0:
            array = array / np.max(np.abs(array))
        
        # Whisper input features 생성 (기본 30초 chunk로 자동 패딩)
        inputs = processor(
            array, 
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # 텍스트 토큰화 (최신 방식)
        labels = processor.tokenizer(
            batch["normalized_text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=448  # Whisper 표준 길이
        ).input_ids[0]
        
        # 🔧 Whisper 전용 형식으로 반환 (input_ids 제외)
        return {
            "input_features": inputs.input_features[0],
            "labels": labels
        }
    
    print("Whisper 데이터셋 전처리 중...")
    # 🔧 컬럼명을 명시적으로 지정하여 오류 방지
    original_columns = train_dataset.column_names
    train_dataset = train_dataset.map(
        prepare_dataset_for_whisper, 
        remove_columns=original_columns
    )
    valid_dataset = valid_dataset.map(
        prepare_dataset_for_whisper, 
        remove_columns=original_columns
    )
    
    # 30초 이하 필터링
    def filter_audio_length(example):
        input_features = example["input_features"]
        
        # input_features가 list인지 tensor인지 확인
        if isinstance(input_features, list):
            # list인 경우, 마지막 차원의 길이를 가져옴
            audio_length = len(input_features) if len(input_features) > 0 else 0
            if audio_length > 0 and hasattr(input_features[0], '__len__'):
                # 2D list의 경우 마지막 차원
                audio_length = len(input_features[0]) if len(input_features[0]) > 0 else audio_length
        else:
            # tensor인 경우
            import torch
            import numpy as np
            if isinstance(input_features, (torch.Tensor, np.ndarray)):
                audio_length = input_features.shape[-1]
            else:
                # 기타 경우 변환 시도
                try:
                    if hasattr(input_features, 'shape'):
                        audio_length = input_features.shape[-1]
                    else:
                        audio_length = len(input_features)
                except:
                    audio_length = 0
        
        return audio_length <= 3000  # Whisper의 최대 길이
    
    train_dataset = train_dataset.filter(filter_audio_length)
    valid_dataset = valid_dataset.filter(filter_audio_length)
    
    print(f"필터링 후 - 학습: {len(train_dataset)}, 검증: {len(valid_dataset)}")
    print_gpu_memory_usage()
    
    # ===== 4. Whisper 전용 데이터 콜레이터 =====
    @dataclass
    class WhisperDataCollator:
        processor: WhisperProcessor
        
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # input_features 배치 처리
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            
            # labels 배치 처리
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            
            # -100으로 패딩된 부분 마스킹 (손실 계산에서 제외)
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            
            # BOS 토큰이 있다면 제거 (Whisper는 BOS 토큰을 예측하지 않음)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            
            batch["labels"] = labels
            
            # 🔧 Whisper + PEFT 버그 수정: input_ids 제거
            # Whisper는 input_features만 받고 input_ids는 받지 않음
            if "input_ids" in batch:
                del batch["input_ids"]
            
            return batch
    
    data_collator = WhisperDataCollator(processor=processor)
    
    # ===== 5. Whisper 평가 메트릭 =====
    def compute_whisper_metrics(pred):
        try:
            import evaluate
            wer_metric = evaluate.load("wer")
            
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            
            # -100을 패드 토큰으로 변경
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            
            # 디코딩
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            
            # WER 계산
            wer = wer_metric.compute(predictions=pred_str, references=label_str)
            
            return {"wer": wer}
        except Exception as e:
            print(f"메트릭 계산 오류: {e}")
            return {}
    
    # ===== 6. Whisper 학습 콜백 =====
    class WhisperTrainingCallback(TrainerCallback):
        """Whisper 전용 학습 콜백"""
        
        def __init__(self):
            self.best_wer = float('inf')
            self.patience = 0
            self.max_patience = 3
            
        def on_evaluate(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
                
            current_wer = logs.get('eval_wer', float('inf'))
            
            if current_wer < self.best_wer:
                self.best_wer = current_wer
                self.patience = 0
                print(f"🎯 Whisper 새로운 최고 WER: {current_wer:.4f}")
                
                # Whisper Turbo Korean 목표 확인
                if current_wer < 0.10:  # ghost613 모델은 4.89% 달성
                    print(f"🎉 우수한 성능 달성! WER {current_wer:.4f} < 0.10")
                elif current_wer < 0.20:
                    print(f"🎉 좋은 성능 달성! WER {current_wer:.4f} < 0.20")
            else:
                self.patience += 1
                print(f"⏳ Early stopping: {self.patience}/{self.max_patience}")
                
            if self.patience >= self.max_patience:
                control.should_training_stop = True
                print("⏹️ Early stopping 활성화")
        
        def on_epoch_end(self, args, state, control, **kwargs):
            # 적극적인 GPU 메모리 정리
            clear_gpu_memory()
            print(f"🧹 에포크 {state.epoch} 완료, GPU 메모리 정리됨")
        
        def on_step_end(self, args, state, control, **kwargs):
            # 매 50 스텝마다 메모리 정리
            if state.global_step % 50 == 0:
                clear_gpu_memory()
        
        def on_train_end(self, args, state, control, **kwargs):
            print(f"📊 Whisper 학습 완료 - 최고 WER: {self.best_wer:.4f}")
    
    # ===== 7. 트레이너 설정 =====
    print("\n7. Whisper Turbo Korean 트레이너 설정...")
    
    training_args = TrainingArguments(
        output_dir="./whisper-turbo-korean-fine-tuned",
        
        # 🔧 PEFT + 4bit: 극도의 메모리 절약
        per_device_train_batch_size=1,      # 최소 배치 크기
        per_device_eval_batch_size=1,       # 최소 배치 크기  
        gradient_accumulation_steps=32,     # 16 → 32 (유효 배치: 32)
        
        # 🔧 학습 기간 설정
        num_train_epochs=5,                 # Whisper는 빠르게 수렴
        max_steps=1500,                     # 안전장치
        
        # 🔧 평가 및 저장 주기
        eval_strategy="steps",
        eval_steps=150,                     # 더 자주 평가
        save_steps=300,
        logging_steps=50,
        
        # 🔧 학습률 설정 (Whisper 권장사항)
        learning_rate=1e-5,                 # Whisper는 낮은 학습률 사용
        lr_scheduler_type="linear",         # Linear decay
        warmup_steps=500,                   # Warmup 중요
        weight_decay=0.01,
        
        # 🔧 적극적인 메모리 최적화 
        fp16=False,                         # FP16 오류 방지를 위해 비활성화
        bf16=bf16_supported,                # bf16 사용 (지원하는 경우만)
        dataloader_pin_memory=False,        # 메모리 절약
        dataloader_num_workers=0,           # CPU 메모리 절약
        gradient_checkpointing=True,        # 메모리 절약 (백프롭 시 메모리 재계산)
        max_grad_norm=1.0,                  # Gradient clipping
        
        # 🔧 추가 메모리 최적화 설정
        ddp_find_unused_parameters=False,   # 사용하지 않는 파라미터 검색 비활성화
        save_safetensors=True,              # 안전한 텐서 저장
        logging_nan_inf_filter=True,        # NaN/Inf 필터링으로 메모리 절약
        
        # 🔧 기타 설정
        save_total_limit=2,                 # 3 → 2 (디스크 절약)
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=None,
        disable_tqdm=False,
        group_by_length=False,              # 🔧 Whisper에서는 input_ids가 없어서 비활성화
        prediction_loss_only=False,
        remove_unused_columns=True,         # 🔧 불필요한 컬럼 제거하여 input_ids 오류 방지
        
        # 🔧 Whisper 특화 설정
        dataloader_drop_last=True,
        no_cuda=False,
        local_rank=-1,
    )
    
    # ===== 8. 콜백 설정 =====
    whisper_callback = WhisperTrainingCallback()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_whisper_metrics,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=processor,     # 🔧 최신 방식: processing_class 사용
        callbacks=[whisper_callback]
    )
    
    print("✅ Whisper Turbo Korean 트레이너 설정 완료")
    print(f"🔧 Whisper 설정 (PEFT + LoRA):")
    print(f"  - 모델: ghost613/whisper-large-v3-turbo-korean (4bit + LoRA)")
    print(f"  - 배치 크기: 1 × 32 = 32 (유효)")
    print(f"  - 에포크: 5")
    print(f"  - 학습률: 1e-5 (Linear decay)")
    print(f"  - 최대 오디오 길이: 20초")
    print(f"  - 목표: WER < 0.10 (우수), WER < 0.20 (좋음)")
    print(f"  - 정밀도: bfloat16 (4bit 양자화)")
    print(f"  - 메모리 최적화: 4bit 양자화 + LoRA + gradient checkpointing")
    
    # ===== 9. Whisper 학습 시작 =====
    print("\n9. Whisper Turbo Korean 학습 시작...")
    print_gpu_memory_usage()
    
    start_time = datetime.now()
    print(f"시작 시간: {start_time}")
    
    try:
        trainer.train()
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\n✅ Whisper Turbo Korean 학습 완료!")
        print(f"총 시간: {duration}")
        
        # 🔧 PEFT 모델 저장 경로 (LoRA 어댑터만 저장)
        model_save_path = "../../models/whisper-turbo-korean-lora"
        os.makedirs("../../models", exist_ok=True)
        
        # LoRA 어댑터만 저장 (원본 모델은 그대로 유지)
        model.save_pretrained(model_save_path)
        processor.save_pretrained(model_save_path)
        print(f"✅ Whisper LoRA 어댑터 저장 완료: {model_save_path}")
        print(f"📌 원본 모델: {MODEL_ID}")
        print(f"📌 LoRA 어댑터: {model_save_path}")
        
        # 최종 평가
        eval_results = trainer.evaluate()
        final_wer = eval_results.get('eval_wer', 'N/A')
        best_wer = whisper_callback.best_wer
        
        print(f"🎯 최종 WER: {final_wer:.4f}")
        print(f"🎯 최고 WER: {best_wer:.4f}")
        
        # ghost613 원본 모델과 비교
        original_wer = 0.0489  # ghost613 모델의 test WER
        print(f"📊 ghost613 원본 WER: {original_wer:.4f}")
        
        if isinstance(best_wer, float):
            if best_wer < 0.05:
                print("🎉 ghost613 원본 모델 수준 달성!")
            elif best_wer < 0.10:
                print("🎉 우수한 성능 달성! (WER < 10%)")
            elif best_wer < 0.20:
                print("🎉 좋은 성능 달성! (WER < 20%)")
            else:
                print("⚠️ 추가 학습 또는 하이퍼파라미터 조정 필요")
            
            # Wav2Vec2 모델들과 비교
            print(f"\n📈 성능 비교:")
            print(f"  - Whisper Turbo Korean: {best_wer:.4f}")
            print(f"  - Phase 2-1 (Wav2Vec2): 0.6321")
            print(f"  - Phase 2-2.5 (Wav2Vec2): 0.6092")
            
            if best_wer < 0.6092:
                improvement = ((0.6092 - best_wer) / 0.6092) * 100
                print(f"  🚀 최고 Phase 대비 {improvement:.1f}% 개선!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 🚀 Whisper Turbo Korean 실험 완료 ===")

# ===== 10. 추론 테스트 함수 =====
def test_whisper_inference(model_path, audio_file_path):
    """Whisper 모델 추론 테스트"""
    print(f"\n🔧 Whisper 추론 테스트: {audio_file_path}")
    
    # 모델 로드
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)
    
    # 오디오 로드 및 처리
    import librosa
    audio, sr = librosa.load(audio_file_path, sr=16000)
    
    # 30초 제한
    if len(audio) > 30 * 16000:
        audio = audio[:30 * 16000]
    
    # 입력 특성 생성
    input_features = processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features
    
    # GPU로 이동 (사용 가능한 경우)
    if torch.cuda.is_available():
        model = model.to("cuda")
        input_features = input_features.to("cuda")
    
    # 추론
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    
    # 디코딩
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    print(f"📝 Whisper 인식 결과: {transcription}")
    return transcription

if __name__ == "__main__":
    main()