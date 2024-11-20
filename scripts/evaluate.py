import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os

# оценка качества модели
def evaluate_model():
    # 1. Чтение гиперпараметров из файла params.yaml
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)
    scoring_params = params.get('scoring', ['f1', 'roc_auc'])
    cv_params = params.get('cv_params', {})
    
    # 2. Загрузка данных и модели
    data = pd.read_csv('data/initial_data.csv')
    model = joblib.load('models/fitted_model.pkl')
    
    # 3. Разделение данных
    X = data.drop(columns=['target'])
    y = data['target']
    
    # 4. Настройка кросс-валидации
    cv_strategy = StratifiedKFold(**cv_params)

    # 5. Оценка модели с кросс-валидацией
    cv_res = cross_validate(
        model,
        X,
        y,
        cv=cv_strategy,
        n_jobs=-1,
        scoring=scoring_params
    )

    # 6. Сохранение всех результатов
    results = {key: round(value.mean(), 3) for key, value in cv_res.items()}
    os.makedirs('cv_results', exist_ok=True)
    with open('cv_results/cv_res.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print("Кросс-валидация завершена. Результаты сохранены в 'cv_results/cv_res.json'.")

if __name__ == '__main__':
    evaluate_model()