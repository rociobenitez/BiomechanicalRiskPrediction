#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importaciones necesarias
import sys
import os
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor

# Asegurar la accesibilidad a módulos personalizados
sys.path.append('../src')

# Importar funciones utilitarias y de modelado
from utils.utils import load_data
from production.model_deployment import (
    split_data,
    scale_data,
    train_random_forest,
    train_gradient_boosting,
    train_extra_trees,
    train_and_evaluate_stacking,
    save_model
)

def main():
    # Ruta para guardar los modelos entrenados
    MODEL_DIR = './models/age_prediction_models'
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Cargar y preparar datos
    df = load_data('../data/processed/prediction_edad_marcha.csv')
    cols = ['edad', 'step length_walk', 'total force rate_walk', 'footstrike type_walk', 'stance excursion (mp->to)_walk', 'imc']
    X_train, X_test, y_train, y_test = split_data(df[cols], 'edad')
    (X_train_ss, X_test_ss), (X_train_mm, X_test_mm) = scale_data(X_train, X_test)
    
    # Entrenar y guardar modelos individuales
    rf_model = train_random_forest(X_train, y_train)
    save_model(rf_model, 'rf_model.pkl', MODEL_DIR)
    
    gbr_model = train_gradient_boosting(X_train, y_train)
    save_model(gbr_model, 'gbr_model.pkl', MODEL_DIR)
    
    et_model = train_extra_trees(X_train, y_train)
    save_model(et_model, 'et_model.pkl', MODEL_DIR)
    
    # Configuración y evaluación de Stacking Regressor con distintos estimadores finales
    final_estimators = [
        ('Linear Regression', LinearRegression()),
        ('Ridge', Ridge()),
        ('ElasticNet', ElasticNet()),
        ('Decision Tree', DecisionTreeRegressor(max_depth=5)),
        ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100))
    ]
    
    base_estimators = [('rf', rf_model),('gbr', gbr_model),('et', et_model)]
    for name, estimator in final_estimators:
        train_and_evaluate_stacking(X_train, y_train, X_test, y_test, base_estimators, estimator, name, MODEL_DIR)

if __name__ == "__main__":
    main()

