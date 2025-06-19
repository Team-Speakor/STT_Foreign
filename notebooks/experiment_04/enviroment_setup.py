import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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