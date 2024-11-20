import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier
import yaml
import os
import joblib

# обучение модели
def fit_model():
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)
    model_params = params.get('model_params', {})
    preprocessor_params = params.get('preprocessor_params', {})
    
	# загрузите результат предыдущего шага: inital_data.csv
    data = pd.read_csv('data/initial_data.csv')
	# реализуйте основную логику шага с использованием гиперпараметров
    cat_features = data.select_dtypes(include='object')
    potential_binary_features = cat_features.nunique() == 2

    binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
    other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
    num_features = data.select_dtypes(['float'])

    # 3. Создание препроцессора
    preprocessor = ColumnTransformer(
        transformers=[
            ('binary', OneHotEncoder(drop='if_binary', **preprocessor_params.get('binary', {})), binary_cat_features.columns.tolist()),
            ('cat', CatBoostEncoder(**preprocessor_params.get('cat', {})), other_cat_features.columns.tolist()),
            ('num', StandardScaler(**preprocessor_params.get('num', {})), num_features.columns.tolist())
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    # 4. Инициализация модели
    model = CatBoostClassifier(**model_params)

    # 5. Создание пайплайна
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # 6. Обучение модели
    X = data.drop(columns=['target'])
    y = data['target']
    pipeline.fit(X, y)    
    
	# сохраните обученную модель в models/fitted_model.pkl
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/fitted_model.pkl')
    print("Model saved to 'models/fitted_model.pkl'")    


if __name__ == '__main__':
	fit_model()