import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import yaml
import os
import joblib

# обучение модели
def fit_model():
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)
    model_params = params.get('model_params', {})
    preprocessor_params = params.get('preprocessor_params', {})
    target_col = params.get('target_col', 'target')
    index_col = params.get('index_col', None)
    
    # загрузите результат предыдущего шага: inital_data.csv
    data = pd.read_csv('data/initial_data.csv')
    
    # Удаляем индексный столбец, если он есть
    if index_col and index_col in data.columns:
        data = data.drop(columns=[index_col])
    
    # Разделяем признаки и целевую переменную
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Выделяем категориальные и числовые признаки
    cat_features = X.select_dtypes(include='object').columns.tolist()
    num_features = X.select_dtypes(include=['float', 'int']).columns.tolist()
    
    # Создание препроцессора
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(**preprocessor_params.get('cat', {})), cat_features),
            ('num', StandardScaler(**preprocessor_params.get('num', {})), num_features)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    # Инициализация модели
    model = LogisticRegression(**model_params)
    
    # Создание пайплайна
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Обучение модели
    pipeline.fit(X, y)    
    
    # Сохранение обученной модели
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/fitted_model.pkl')
    print("Model saved to 'models/fitted_model.pkl'")    

if __name__ == '__main__':
    fit_model()