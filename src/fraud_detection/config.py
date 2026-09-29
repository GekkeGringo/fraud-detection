from pathlib import Path

# Корень проекта — папка на два уровня выше этого файла
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_PATH = PROJECT_ROOT / 'data' / 'creditcard.csv'
MODELS_DIR = PROJECT_ROOT / 'models'
MODEL_PATH = MODELS_DIR / 'model.joblib'

RANDOM_STATE = 42
TEST_SIZE = 0.2