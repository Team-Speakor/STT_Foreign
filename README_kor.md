# STT_Foreign: 외국인 한국어 STT 모델 학습

[![Project Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)](https://github.com/Team-Speakor/STT_Foreign)
[![License](https://img.shields.io/badge/License-Educational-orange?style=flat-square)]()
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)](https://www.python.org/)
[![Team](https://img.shields.io/badge/Team-Hanyang%20ERICA-blue?style=flat-square)](https://www.hanyang.ac.kr/)

## 프로젝트 개요

이 레포지토리는 **Speakor** 프로젝트의 핵심 구성요소 중 **외국인 한국어 STT 모델 학습** 부분을 담당합니다. 

> **Speakor**는 외국인 한국어 학습자의 발음 오류를 탐지하고 피드백을 제공하는 AI 기반 음성 교정 시스템으로, 이중 STT 모델 구조를 사용하여 외국인용 STT와 한국인용 STT 결과를 비교 분석합니다.

### 이 레포지토리의 역할

본 레포지토리는 **외국인 화자의 비표준 발화를 표준 한국어 텍스트로 변환하는 STT 모델**을 학습하고 최적화하는 과정을 담고 있습니다.

- **모델**: Wav2Vec2 기반 외국인 한국어 음성 인식
- **데이터**: AI Hub 외국인 한국어 발화 음성 데이터 (~20,000개 문장)
- **목표**: 외국인 발음을 정확한 한국어 텍스트로 변환하는 고성능 STT 모델 개발

## 전체 시스템에서의 위치

본 STT_Foreign 모델은 Speakor 시스템의 핵심 구성요소로, **4개 레포지토리 중 하나**입니다:

![System Diagram](./images/System_Diagram.png)

```
전체 시스템 구조:
├── STT_Foreign (이 레포지토리) → 외국인 음성 → 표준 텍스트
├── STT_Korean → 동일 음성 → 실제 청취 텍스트  
├── Server → 백엔드 API, 오류 분석, GPT 피드백
└── Client_React → 프론트엔드 UI, 결과 시각화
```

### 관련 레포지토리:
- 🔸 **[STT_Foreign](https://github.com/Team-Speakor/STT_Foreign)** - 외국인 STT 모델 (이 레포지토리)
- **[STT_Korean](https://github.com/Team-Speakor/STT_Korean)** - 한국인 STT 모델
- **[Server](https://github.com/Team-Speakor/Server)** - FastAPI 백엔드
- **[Client_React](https://github.com/Team-Speakor/Client_React)** - React 프론트엔드

### 처리 플로우에서의 역할:

![User Flow Diagram](./images/Userflow_Diagram.png)

```
[음성 입력] → [화자 분리] → [이중 STT 추론]
                                    ├── 🔸 STT_Foreign (이 레포지토리)
                                    └── STT_Korean
                                    ↓
                              [오류 분석 & 피드백] (Server)
                                    ↓
                              [결과 시각화] (Client_React)
```

### 기술 스택 (이 레포지토리)
| 구성요소 | 기술 | 역할 |
|----------|------|------|
| **STT 모델** | Wav2Vec2 (HuggingFace) | 외국인 음성 → 표준 한국어 텍스트 |
| **데이터 처리** | pandas, librosa | 음성 데이터 전처리 및 분석 |
| **모델 학습** | transformers, torch | 모델 파인튜닝 및 평가 |
| **평가 지표** | WER, CER | 음성 인식 성능 측정 |

## 데이터셋

### 사용 데이터
- **외국인 한국어 발화 음성 데이터**: AI Hub 제공, 약 20,000개 문장
  - 다양한 모국어 배경 (중국어, 영어, 일본어 등)
  - 발음 오류, 억양 왜곡 등 외국인 화자 특성 포함
  - **용도**: 본 레포지토리의 외국인 STT 모델 학습

### 데이터 전처리
- 텍스트 정규화: 한글만 추출, 공백 정리
- 오디오 정규화: 16kHz 샘플링, 볼륨 정규화
- 라벨 누락 데이터 필터링

## 모델 학습 및 실험 과정

### Experiment Timeline

#### **프로젝트 기반 구축** (민경진)
- **데이터 발굴**: AI Hub에서 외국인 한국어 발화 데이터 발굴 및 확보
- **시스템 아키텍처 설계**: 이중 STT 모델 비교 기반 발음 오류 탐지 아이디어 제안
- **전체 플로우 설계**: 음성 → STT → 비교 → 피드백 파이프라인 설계

#### Experiment 01: 초기 모델 구축 (김용재)
- **목표**: 기본 Wav2Vec2 모델 파인튜닝 시작
- **핵심 파일**: `notebooks/experiment_01/`

#### Experiment 02: 모델 최적화 시도 (민경진)
- **목표**: 학습 안정성 개선 및 성능 향상
- **주요 기여**: 데이터 전처리 파이프라인 구축, 학습 설정 실험
- **핵심 파일**: `notebooks/experiment_02/combine_emergency_fixed.py`

#### Experiment 03: Whisper 모델 도전 (민경진)
- **혁신 시도**: 최신 Whisper 모델 적용으로 성능 향상 도전
- **모델**: `ghost613/whisper-large-v3-turbo-korean`
- **기술적 도전**: 
  - PEFT (LoRA) 기반 4bit 양자화 학습
  - 메모리 최적화 설정 적용
  - 582줄의 정교한 학습 코드 작성
- **결과**: GPU 메모리 한계로 인한 기술적 제약 확인
- **의의**: 최신 기술 적용 가능성 탐색, 향후 발전 방향 제시
- **핵심 파일**: `notebooks/experiment_03/whisper_turbo_korean_train.py`

#### Experiment 04-05: 최종 모델 완성 (김용재)
- **독립적 발전**: 이전 실험들을 바탕으로 새로운 최적화 전략 개발
- **핵심 혁신**:
  - Feature Encoder 5 에포크 후 점진적 해제 전략 도입
  - Gradient Accumulation 8 스텝으로 학습 안정성 확보
  - 학습률 3e-4 고정으로 최적 성능 달성
- **최종 성과**:
  - **WER**: 0.186573 (18.66%) - 이전 실험 대비 대폭 개선
  - **CER**: 0.073379 (7.34%)
- **핵심 파일**: `notebooks/experiment_05/`

### 모델 설정 비교

| 실험 | 담당자 | 모델 | 주요 기여 | 결과 |
|------|--------|------|-----------|------|
| **프로젝트 기획** | 민경진 | - | 데이터 발굴, 시스템 아이디어 | 프로젝트 방향 제시 |
| **Experiment 02** | 민경진 | Wav2Vec2 | 초기 학습 시도 | 경험과 인사이트 축적 |
| **Experiment 03** | 민경진 | Whisper-Turbo | 혁신 기술 도전 | 기술 가능성 탐색 |
| **Experiment 01,04,05** | 김용재 | Wav2Vec2 | 최적화 전략 개발 | **WER 0.1866 달성** |

## Quick Start

### Prerequisites
```bash
# Python 3.8+ 필요
pip install torch transformers librosa pandas numpy
```

### 모델 사용법
```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa

# 최종 모델 로드 (Google Drive에서 다운로드 필요)
model = Wav2Vec2ForCTC.from_pretrained("./models/phase2-2-6-conservative-model/")
processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

# 음성 파일 추론
def transcribe_audio(audio_path):
    speech, rate = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    
    return transcription

# 사용 예시
result = transcribe_audio("sample_audio.wav")
print(f"인식 결과: {result}")
```

## 최종 성과

### 모델 성능 지표
| 지표 | 테스트셋 | 실제 환경 |
|------|----------|-----------|
| **WER** | **18.66%**  | 29-31% |
| **CER** | **7.34%** | - |
| **개선율** | **67%**  | - |

### 프로젝트 성과
- **목표 달성**: 외국인 STT 모델 성공적 구축
- **성능 개선**: 초기 대비 67% 향상된 안정적 모델
- **기술 혁신**: Whisper 모델 적용으로 미래 방향 제시  

## 레포지토리 구조

```
STT_Foreign/
├── data/                          # 학습 데이터
│   ├── train/                     # 학습용 음성 및 라벨
│   ├── valid/                     # 검증용 음성 및 라벨
│   └── test/                      # 테스트용 데이터
├── notebooks/                     # 실험 노트북 모음
│   ├── experiment_01/             # 초기 실험 (김용재)
│   ├── experiment_02/             # Wav2Vec2 개선 시도 (민경진)
│   │   ├── combine_emergency_fixed.py
│   ├── experiment_03/             # Whisper 시도 (민경진)
│   │   ├── whisper_turbo_korean_train.py
│   ├── experiment_04/             # 중간 최적화 (김용재)
│   └── experiment_05/             # 🎯 최종 모델 (김용재)
│       ├── combine.ipynb          # 최종 학습 코드
│       ├── model_finetuning.py    # 모델 파인튜닝
│       ├── data_preprocessing_and_eda.py
│       └── common_utils.py        # 유틸리티 함수
├── models/                        # 학습된 모델 저장소
│   ├── phase2-2-5-model/          # 중간 모델
│   ├── phase2-2-5-training-log.csv
│   └── phase2-2-6-conservative-model/
├── results/                       # 실험 결과
│   └── result.txt                 # 최종 WER/CER 결과
└── README.md                      # 프로젝트 문서
```

### 주요 파일 설명

#### **experiment_05/** (최종 완성 모델)
- `combine.ipynb`: 최종 학습 코드 및 결과
- `model_finetuning.py`: 모델 파인튜닝 스크립트
- `data_preprocessing_and_eda.py`: 데이터 전처리 로직

#### **experiment_03/** (혁신 기술 도전)
- `whisper_turbo_korean_train.py`: 최신 Whisper 모델 적용 도전
- 4bit 양자화 + PEFT LoRA 최신 기법 적용
- GPU 메모리 최적화 기법 다수 적용
- 향후 기술 발전 방향 제시

#### **results.txt**
- `result.txt`: 최종 WER 0.186573, CER 0.073379

## 기술적 도전과 해결

### 1. Whisper 모델 적용 시도 (민경진)
**도전 배경**: 최신 Whisper 모델로 성능 향상
```python
# 4bit 양자화 + PEFT LoRA 최신 기법 적용
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# LoRA 설정으로 효율적 파인튜닝
lora_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="SEQ_2_SEQ_LM"
)
```
**의의**: 
- GPU 메모리 한계 확인으로 현실적 제약 파악
- 향후 발전 방향과 기술 로드맵 제시

### 2. 학습 성능 최적화
**문제**: 초기 실험에서 WER 60% 이상의 낮은 성능
**해결책**:
- Gradient Accumulation Steps 1 → 8로 증가
- 학습률 스케줄링 제거, 고정 학습률 사용
- Feature Encoder 점진적 해제 전략 도입
**결과**: WER 18.66%까지 개선

## 성과 지표

![WER Score](https://img.shields.io/badge/WER-18.66%25-success?style=flat-square)
![CER Score](https://img.shields.io/badge/CER-7.34%25-success?style=flat-square)
![Performance Improvement](https://img.shields.io/badge/성능개선-67%25-brightgreen?style=flat-square)
![Data Size](https://img.shields.io/badge/데이터-20K문장-blue?style=flat-square)

## 기술 스택

### 모델 및 프레임워크
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow?style=flat-square)
![Wav2Vec2](https://img.shields.io/badge/Model-Wav2Vec2-blue?style=flat-square)

### 실험 및 분석
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Librosa](https://img.shields.io/badge/Audio-Librosa-green?style=flat-square)

## 개발자 정보

**전체 Speakor 팀**: 음정우, 김용재, 민경진, 박지우, 이유준 (한양대학교 ERICA 컴퓨터공학과)

### 주요 기여자

<table>
    <tr height="160px">
        <td align="center" width="160px">
            <a href="https://github.com/KJ-Min"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/KJ-Min"/></a>
            <br/>
            <a href="https://github.com/KJ-Min"><strong>민경진</strong></a>
            <br />
            <sub>Project Designer</sub>
        </td>
        <td>
            <strong>🎯 프로젝트 기획 & 기술 탐색</strong><br/>
            • AI Hub 데이터 발굴 및 시스템 아키텍처 설계<br/>
            • 이중 STT 모델 비교 기반 발음 오류 탐지 아이디어 제안<br/>
            • Whisper 모델 적용 시도 (PEFT LoRA + 4bit 양자화)<br/>
            • Experiment 02, 03 주도 - 혁신 기술 도전<br/>
            • 전체 시스템 플로우 설계 및 파이프라인 구축
        </td>
    </tr>
</table>

<table>
    <tr height="160px">
        <td align="center" width="160px">
            <a href="https://github.com/Songforthesilent"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/Songforthesilent"/></a>
            <br/>
            <a href="https://github.com/Songforthesilent"><strong>김용재</strong></a>
            <br />
            <sub>Lead Engineer</sub>
        </td>   
        <td>
            <strong>🚀 모델 최적화 & 성과 달성</strong><br/>
            • <b>최종 WER 18.66% 달성</b> - 프로젝트 핵심 성과 🏆<br/>
            • Wav2Vec2 체계적 최적화 및 하이퍼파라미터 튜닝<br/>
            • Feature Encoder 점진적 해제 전략 개발<br/>
            • Experiment 01, 04, 05 완성 - 안정적 모델 구현<br/>
            • 67% 성능 개선으로 프로젝트 목표 달성
        </td> 
    </tr>
</table>

## 참고자료

### 전체 프로젝트
- **GitHub Organization**: [Team-Speakor](https://github.com/Team-Speakor)
  - **[STT_Foreign](https://github.com/Team-Speakor/STT_Foreign)**: 외국인 STT 모델 (이 레포지토리)
  - **[STT_Korean](https://github.com/Team-Speakor/STT_Korean)**: 한국인 STT 모델
  - **[Server](https://github.com/Team-Speakor/Server)**: 백엔드 서버 (FastAPI)
  - **[Client_React](https://github.com/Team-Speakor/Client_React)**: 프론트엔드 (React)

### 프로젝트 문서
- **최종 보고서**: 상세한 시스템 설계 및 구현 결과
- **Notion 문서**: [프로젝트 진행 기록](https://agate-pulsar-e23.notion.site/1bbfac7f80918021bd60f6707c6d689c)
- **모델 가중치**: [Google Drive 링크](https://drive.google.com/drive/folders/1Qf9Ckv8WvzoYZ_QaQY9v1ibdcst6S4H6?usp=sharing)

### 데이터셋 및 기술
- [AI Hub - 외국인 한국어 발화 음성 데이터](https://aihub.or.kr/)
- [kresnik/wav2vec2-large-xlsr-korean](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean) - 베이스 모델
- [ghost613/whisper-large-v3-turbo-korean](https://huggingface.co/ghost613/whisper-large-v3-turbo-korean) - Whisper 시도용
