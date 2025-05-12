# STT_Foreign
 <details> <summary><code>STT_Foreign/</code> 폴더 구조 펼치기/접기</summary> 
 STT_Foreign/
├── data/                     # 데이터 관련 폴더
│   ├── train/                # 학습 데이터
│   ├── validation/           # 검증 데이터
│   └── test/                 # 테스트 데이터
├── notebooks/                # Jupyter Notebooks 
│   ├── 00_environment_setup.ipynb          # 환경 설정 및 라이브러리 설치
│   ├── 01_data_preprocessing_and_eda.ipynb # 데이터 전처리 및 EDA
│   ├── 02_model_finetuning.ipynb           # 모델 미세조정
│   ├── 03_model_evaluation.ipynb           # 모델 평가
│   └── common_utils.py                     # 노트북 간 공유 함수 모음 
├── models/                   # 학습된 모델 파일 및 설정 저장
│   ├── experiment_01/
│   │   ├── model_weights.pth
│   │   ├── config.json
│   │   └── training_log.txt
│   └── ...
├── results/                  # 모델 평가 결과 및 분석 자료
│   ├── experiment_01_evaluation.json
│   └── ...
├── configs/                  # 프로젝트 전반의 주요 설정 파일
│   ├── data_config.yaml
│   └── train_config.yaml
├── requirements.txt          # Python 패키지 의존성 목록 
└── README.md                 # 프로젝트 설명, 실행 방법 등
 </details> 
 폴더/파일 설명

    data/
      ├─ train/: 학습 데이터
      ├─ validation/: 검증 데이터
      └─ test/: 테스트 데이터

    notebooks/
      ├─ 00_environment_setup.ipynb: 환경 설정 및 라이브러리 설치
      ├─ 01_data_preprocessing_and_eda.ipynb: 데이터 전처리 및 EDA
      ├─ 02_model_finetuning.ipynb: 모델 미세조정
      ├─ 03_model_evaluation.ipynb: 모델 평가
      └─ common_utils.py: 노트북 간 공유 함수 모음

    models/
      ├─ 학습된 모델 가중치, 설정, 로그 등 저장

    results/
      ├─ 모델 평가 결과 및 분석 자료

    configs/
      ├─ 데이터 및 학습 관련 주요 설정 파일

    requirements.txt
      ├─ Python 패키지 의존성 목록

    README.md
      ├─ 프로젝트 설명, 실행 방법 등

