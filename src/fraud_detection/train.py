import joblib
import lightgbm as lgb

from fraud_detection.config import MODELS_DIR, MODEL_PATH, RANDOM_STATE
from fraud_detection.data import load_data, split_data


def train_model(X_train, y_train) -> lgb.LGBMClassifier:
    """Обучает baseline-модель LightGBM с балансировкой классов."""
    model = lgb.LGBMClassifier(
        random_state=RANDOM_STATE,
        class_weight='balanced',
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def main():
    print('Загрузка данных...')
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    print('Обучение модели...')
    model = train_model(X_train, y_train)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f'Модель сохранена: {MODEL_PATH}')


if __name__ == '__main__':
    main()