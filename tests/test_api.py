from unittest.mock import patch

from fastapi.testclient import TestClient

import fraud_detection.api as api_module


def make_client(fake_model):
    with patch.object(api_module, 'model', fake_model):
        yield TestClient(api_module.app)


def test_health_check(fake_model):
    client = TestClient(api_module.app)
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_predict_returns_valid_response(fake_model):
    with patch.object(api_module, 'model', fake_model):
        client = TestClient(api_module.app)
        sample_transaction = {
            'Time': 0.0, 'V1': -1.36, 'V2': -0.07, 'V3': 2.54, 'V4': 1.38,
            'V5': -0.34, 'V6': 0.46, 'V7': 0.24, 'V8': 0.10, 'V9': 0.36,
            'V10': 0.09, 'V11': -0.55, 'V12': -0.62, 'V13': -0.99, 'V14': -0.31,
            'V15': 1.47, 'V16': -0.47, 'V17': 0.21, 'V18': 0.03, 'V19': 0.40,
            'V20': 0.25, 'V21': -0.02, 'V22': 0.28, 'V23': -0.11, 'V24': 0.07,
            'V25': 0.13, 'V26': -0.19, 'V27': 0.13, 'V28': -0.02, 'Amount': 149.62
        }

        response = client.post('/predict', json=sample_transaction)

        assert response.status_code == 200
        data = response.json()
        assert 'fraud_probability' in data
        assert 'is_fraud' in data
        assert 0.0 <= data['fraud_probability'] <= 1.0
        assert isinstance(data['is_fraud'], bool)


def test_predict_rejects_missing_fields(fake_model):
    with patch.object(api_module, 'model', fake_model):
        client = TestClient(api_module.app)
        incomplete_transaction = {'Time': 0.0, 'V1': -1.36}

        response = client.post('/predict', json=incomplete_transaction)

        assert response.status_code == 422