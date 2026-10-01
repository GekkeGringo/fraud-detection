from fraud_detection.data import load_data, split_data


def test_load_data_shape():
    df = load_data()
    assert df.shape[1] == 31
    assert df.shape[0] > 0


def test_load_data_no_missing_values():
    df = load_data()
    assert df.isnull().sum().sum() == 0


def test_split_data_preserves_class_balance():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    train_fraud_rate = y_train.mean()
    test_fraud_rate = y_test.mean()

    assert abs(train_fraud_rate - test_fraud_rate) < 0.001


def test_split_data_sizes():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    assert len(X_train) + len(X_test) == len(df)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)