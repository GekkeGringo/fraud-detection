import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fake_df():
    rng = np.random.default_rng(42)
    n_rows = 1000
    data = {f'V{i}': rng.normal(size=n_rows) for i in range(1, 29)}
    data['Time'] = rng.uniform(0, 172792, size=n_rows)
    data['Amount'] = rng.uniform(0, 500, size=n_rows)

    n_fraud = 20
    classes = np.array([1] * n_fraud + [0] * (n_rows - n_fraud))
    rng.shuffle(classes)
    data['Class'] = classes

    return pd.DataFrame(data)


@pytest.fixture
def fake_model(fake_df):
    X = fake_df.drop(columns=['Class'])
    y = fake_df['Class']
    model = lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
    model.fit(X, y)
    return model