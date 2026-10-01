from fraud_detection.data import split_data


def test_split_data_preserves_class_balance(fake_df):
    X_train, X_test, y_train, y_test = split_data(fake_df)

    train_fraud_rate = y_train.mean()
    test_fraud_rate = y_test.mean()

    assert abs(train_fraud_rate - test_fraud_rate) < 0.02


def test_split_data_sizes(fake_df):
    X_train, X_test, y_train, y_test = split_data(fake_df)

    assert len(X_train) + len(X_test) == len(fake_df)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)