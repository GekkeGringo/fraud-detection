import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from fraud_detection.config import PROJECT_ROOT
from fraud_detection.data import load_data, split_data

REPORTS_DIR = PROJECT_ROOT / 'reports'


def generate_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, output_name: str):
    """Строит и сохраняет HTML-отчёт о дрейфе данных между reference и current."""
    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference_df, current_data=current_df)

    REPORTS_DIR.mkdir(exist_ok=True)
    output_path = REPORTS_DIR / output_name
    result.save_html(str(output_path))
    print(f'Отчёт сохранён: {output_path}')


def main():
    df = load_data()
    X_train, X_test, _, _ = split_data(df)

    print('Строим отчёт: train vs test (ожидаем отсутствие дрейфа)...')
    generate_drift_report(X_train, X_test, 'drift_report_train_vs_test.html')

    print('Строим отчёт с искусственным дрейфом в Amount...')
    X_test_drifted = X_test.copy()
    X_test_drifted['Amount'] = X_test_drifted['Amount'] * 3 + 50
    generate_drift_report(X_train, X_test_drifted, 'drift_report_simulated_drift.html')


if __name__ == '__main__':
    main()