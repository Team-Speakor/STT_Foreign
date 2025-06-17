#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한국어 음성 인식 모델 훈련 스크립트 - Phase 2-2.6-Conservative
GPU OOM 고려한 현실적 개선 - Step 1 (안전 우선)
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
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments, TrainerCallback
from transformers import get_cosine_schedule_with_warmup
from dataclasses import dataclass
from typing import Dict, List, Union
import matplotlib.pyplot as plt

# 🔧 데드락 해결 설정
os.environ["NCCL_P2P_DISABLE"] = "1"  # 🚨 핵심: NCCL P2P 비활성화
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Single GPU 강제 사용
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    print("=== 🚀 Phase 2-2.6-Conservative: GPU OOM 고려한 현실적 개선 ===")
    print("🔧 Step 1: 안전 우선 + 점진적 레이어 해제 + 안정적 학습률")
    
    # GPU 설정 확인
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    print(f"GPU 개수: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"현재 GPU: {torch.cuda.current_device()}")
        print(f"GPU 이름: {torch.cuda.get_device_name()}")
    
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
    
    print(f"🔧 Phase 2-2.6-Conservative: 전체 데이터 사용 - 학습: {len(train_df)}개, 검증: {len(valid_df)}개")
    
    # 텍스트 정규화
    def prepare_korean_text(text):
        text = re.sub(r'[^\uAC00-\uD7A3\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    train_df['normalized_text'] = train_df['text'].apply(prepare_korean_text)
    valid_df['normalized_text'] = valid_df['text'].apply(prepare_korean_text)
    
    print(f"최종 데이터 크기 - 학습: {len(train_df)}, 검증: {len(valid_df)}")
    
    # ===== 2. 모델 로딩 =====
    print("\n2. 모델 로딩...")
    
    MODEL_ID = "kresnik/wav2vec2-large-xlsr-korean"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        gradient_checkpointing=True,  # 🔧 메모리 최적화 추가
        low_cpu_mem_usage=True,
    )
    
    # 🔧 Phase 2-2.6-Conservative: Feature encoder만 고정 (combine.ipynb 방식)
    model.freeze_feature_encoder()
    print("✅ Feature Encoder만 고정됨 (점진적 해제 예정)")
    
    # 🔧 Single GPU 강제 설정
    if torch.cuda.is_available():
        model = model.to("cuda:0")
    
    print("✅ 모델 로딩 완료 (Feature Encoder만 고정, Single GPU)")
    
    # ===== 3. 데이터셋 준비 =====
    print("\n3. 데이터셋 준비...")
    
    def prepare_dataset(df):
        dataset = Dataset.from_pandas(df)
        dataset = dataset.cast_column("file_path", Audio(sampling_rate=16000))
        return dataset
    
    train_dataset = prepare_dataset(train_df)
    valid_dataset = prepare_dataset(valid_df)
    
    def prepare_dataset_for_model_safe(batch):
        audio = batch["file_path"]
        array = audio["array"]
        
        # 🔧 15초 오디오 유지
        max_samples = 15 * 16000
        if len(array) > max_samples:
            array = array[:max_samples]
        
        if np.max(np.abs(array)) > 0:
            array = array / np.max(np.abs(array))
        
        batch["input_values"] = processor(
            array, 
            sampling_rate=16000,
            max_length=max_samples,
            truncation=True,
            padding=False
        ).input_values[0]
        
        with processor.as_target_processor():
            batch["labels"] = processor(batch["normalized_text"]).input_ids
        
        return batch
    
    print("데이터셋 전처리 중...")
    train_dataset = train_dataset.map(prepare_dataset_for_model_safe, remove_columns=train_dataset.column_names)
    valid_dataset = valid_dataset.map(prepare_dataset_for_model_safe, remove_columns=valid_dataset.column_names)
    
    # 15초 필터링
    def filter_15sec(example):
        audio_length = len(example["input_values"])
        return 1*16000 <= audio_length <= 15*16000
    
    train_dataset = train_dataset.filter(filter_15sec)
    valid_dataset = valid_dataset.filter(filter_15sec)
    
    print(f"15초 필터링 후 - 학습: {len(train_dataset)}, 검증: {len(valid_dataset)}")
    
    # ===== 4. 데이터 콜레이터 =====
    @dataclass
    class DataCollatorCTCSafe:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
            
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")

            batch["labels"] = labels_batch["input_ids"]
            return batch
    
    data_collator = DataCollatorCTCSafe(processor=processor)
    
    # ===== 5. 평가 메트릭 =====
    def compute_metrics_safe(pred):
        try:
            import evaluate
            wer_metric = evaluate.load("wer")
            
            pred_logits = pred.predictions
            pred_ids = np.argmax(pred_logits, axis=-1)
            pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
            
            pred_str = processor.batch_decode(pred_ids)
            label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
            
            wer = wer_metric.compute(predictions=pred_str, references=label_str)
            return {"wer": wer}
        except Exception as e:
            print(f"메트릭 계산 오류: {e}")
            return {}
    
    # ===== 6. Phase 2-2.6-Conservative: 스마트 언프리즈 콜백 =====
    class SmartUnfreezeCallback(TrainerCallback):
        """combine.ipynb 방식의 점진적 레이어 해제"""
        
        def on_epoch_begin(self, args, state, control, **kwargs):
            model = kwargs.get('model', None)
            if model is None:
                return
                
            if state.epoch == 5:
                # Feature encoder만 해제 (combine.ipynb 방식)
                model.wav2vec2.feature_extractor._freeze_parameters = False
                for param in model.wav2vec2.feature_extractor.parameters():
                    param.requires_grad = True
                print("🔓 Feature Encoder 해제됨 (5 에포크)")
            elif state.epoch == 8:
                # Conservative에서는 8 에포크에서 상위 2개 Transformer 레이어만 추가 해제
                for layer in model.wav2vec2.encoder.layers[-2:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                print("🔓 상위 2개 Transformer 레이어 해제됨 (8 에포크)")
        
        def on_epoch_end(self, args, state, control, **kwargs):
            # 에포크 끝에 GPU 캐시 정리
            torch.cuda.empty_cache()
            print(f"🧹 에포크 {state.epoch} 완료, GPU 캐시 정리됨")
    
    # ===== 7. Phase 2-2.6-Conservative: 간단한 Early Stopping =====
    class SimpleEarlyStoppingCallback(TrainerCallback):
        """버그 수정된 간단한 Early Stopping"""
        
        def __init__(self, patience=3):
            self.patience = patience
            self.best_wer = float('inf')
            self.wait = 0
            
        def on_evaluate(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
                
            current_wer = logs.get('eval_wer', float('inf'))
            
            if current_wer < self.best_wer:
                self.best_wer = current_wer
                self.wait = 0
                print(f"🎯 새로운 최고 WER: {current_wer:.4f}")
                
                # Phase 2-2.6 목표 확인
                if current_wer < 0.55:
                    print(f"🎉 Phase 2-2.6-Conservative 1차 목표 달성! WER {current_wer:.4f} < 0.55")
                if current_wer < 0.50:
                    print(f"🎉 Phase 2-2.6-Conservative 2차 목표 달성! WER {current_wer:.4f} < 0.50")
            else:
                self.wait += 1
                print(f"⏳ Early stopping: {self.wait}/{self.patience}")
                
            if self.wait >= self.patience:
                control.should_training_stop = True
                print("⏹️ Early stopping 활성화")
        
        def on_train_end(self, args, state, control, **kwargs):
            print(f"📊 학습 완료 - 최고 WER: {self.best_wer:.4f}")
    
    # ===== 8. 트레이너 설정 =====
    print("\n8. Phase 2-2.6-Conservative 트레이너 설정...")
    
    training_args = TrainingArguments(
        output_dir="./wav2vec2-phase2-2-6-conservative",
        # 🔧 메모리 안전 설정
        per_device_train_batch_size=2,      # 유지 (OOM 방지)
        per_device_eval_batch_size=2,       # 유지
        gradient_accumulation_steps=4,      # 1 → 4 (유효 배치: 8)
        
        # 🔧 학습 기간 현실적 조정
        num_train_epochs=8,                 # Conservative: 8 에포크
        max_steps=2000,                     # 안전장치
        
        # 🔧 평가 및 저장 주기
        eval_strategy="steps",
        eval_steps=100,                     # 더 자주 모니터링
        save_steps=200,                     # 체크포인트 빈도 증가
        logging_steps=50,
        
        # 🔧 안정적 학습률 전략
        learning_rate=3e-4,                 # combine.ipynb와 동일
        lr_scheduler_type="constant",       # 스케줄링 제거
        warmup_steps=100,                   # 최소한의 warmup
        weight_decay=0.005,                 # 유지
        
        # 🔧 메모리 최적화
        fp16=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        
        # 🔧 기타 설정
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=None,
        disable_tqdm=False,
        group_by_length=False,
        no_cuda=False,
        local_rank=-1,
        ddp_find_unused_parameters=False,
    )
    
    # ===== 9. 콜백 설정 =====
    smart_unfreeze_callback = SmartUnfreezeCallback()
    early_stopping_callback = SimpleEarlyStoppingCallback(patience=3)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics_safe,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=processor,
        callbacks=[smart_unfreeze_callback, early_stopping_callback]
    )
    
    print("✅ Phase 2-2.6-Conservative 트레이너 설정 완료")
    print(f"🔧 Conservative 설정:")
    print(f"  - 레이어 고정: Feature encoder만 (5 에포크 후 해제)")
    print(f"  - 에포크 수: 8 (안전 우선)")
    print(f"  - Gradient accumulation: 4 (유효 배치: 8)")
    print(f"  - 학습률: 3e-4 (고정, 스케줄링 없음)")
    print(f"  - Early Stopping: 3 patience")
    print(f"  - 목표: WER < 0.55 (1차), WER < 0.50 (2차)")
    
    # ===== 10. 안전한 학습 시작 =====
    print("\n10. Phase 2-2.6-Conservative 학습 시작...")
    
    start_time = datetime.now()
    print(f"시작 시간: {start_time}")
    
    try:
        trainer.train()
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\n✅ Phase 2-2.6-Conservative 학습 완료!")
        print(f"총 시간: {duration}")
        
        # 🔧 모델 저장 경로
        model_save_path = "../../models/phase2-2-6-conservative-model"
        os.makedirs("../../models", exist_ok=True)
        
        model.save_pretrained(model_save_path)
        processor.save_pretrained(model_save_path)
        print(f"✅ 모델 저장 완료: {model_save_path}")
        
        # 최종 평가
        eval_results = trainer.evaluate()
        final_wer = eval_results.get('eval_wer', 'N/A')
        best_wer = early_stopping_callback.best_wer
        
        print(f"🎯 최종 WER: {final_wer:.4f}")
        print(f"🎯 최고 WER: {best_wer:.4f}")
        
        # Phase 2-2.6-Conservative 목표 달성 확인
        if isinstance(best_wer, float):
            if best_wer < 0.50:
                print("🎉 Phase 2-2.6-Conservative 2차 목표 달성! WER < 0.50")
                print("✅ Phase 2-2.6-Aggressive 진행 조건 만족")
            elif best_wer < 0.55:
                print("🎉 Phase 2-2.6-Conservative 1차 목표 달성! WER < 0.55")
                print("✅ 다음 단계 진행 가능")
            else:
                print("⚠️ Phase 2-2.6-Conservative 목표 미달성")
                print("🔧 하이퍼파라미터 조정 또는 데이터 분석 필요")
            
            # 이전 Phase들과 비교
            phase_2_1_wer = 0.6321
            phase_2_2_5_wer = 0.6092
            
            if best_wer < phase_2_1_wer:
                improvement_2_1 = ((phase_2_1_wer - best_wer) / phase_2_1_wer) * 100
                print(f"📈 Phase 2-1 대비 {improvement_2_1:.2f}% 개선")
            
            if best_wer < phase_2_2_5_wer:
                improvement_2_2_5 = ((phase_2_2_5_wer - best_wer) / phase_2_2_5_wer) * 100
                print(f"📈 Phase 2-2.5 대비 {improvement_2_2_5:.2f}% 개선")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 🚀 Phase 2-2.6-Conservative 실험 완료 ===")

if __name__ == "__main__":
    main() 