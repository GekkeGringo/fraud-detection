import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import DATA_PATH, RANDOM_STATE, TEST_SIZE


def load_data() -> pd.DataFrame:
    """Загружает исходный датасет с диска."""
    return pd.read_csv(DATA_PATH)


def split_data(df: pd.DataFrame):
    """Разбивает данные на train/test со стратификацией по классу."""
    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test