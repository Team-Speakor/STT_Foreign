# STT_Foreign
STT_Foreign/
├── data/                     # 데이터 관련 폴더
│   ├── train/                 
│   ├── validation/                
│   └── test/           
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
