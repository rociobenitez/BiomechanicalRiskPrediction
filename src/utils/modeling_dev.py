import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    ExtraTreesRegressor, 
    StackingRegressor
)
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score, 
    f1_score, 
    confusion_matrix, 
    classification_report, 
    classification_report, 
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    root_mean_squared_error,
    roc_curve,
    auc
)


def split_data(df, columna_objetivo, test_size=0.2, random_state=42):
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    X = df.drop(columna_objetivo, axis=1)
    y = df[columna_objetivo]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_data(X_train, X_test):
    """Escala los datos usando StandardScaler y MinMaxScaler."""
    scaler_standard = StandardScaler()
    X_train_ss = pd.DataFrame(scaler_standard.fit_transform(X_train), columns=X_train.columns)
    X_test_ss = pd.DataFrame(scaler_standard.transform(X_test), columns=X_test.columns)

    scaler_minmax = MinMaxScaler(feature_range=(0, 1))
    X_train_mm = pd.DataFrame(scaler_minmax.fit_transform(X_train), columns=X_train.columns)
    X_test_mm = pd.DataFrame(scaler_minmax.transform(X_test), columns=X_test.columns)

    return (X_train_ss, X_test_ss), (X_train_mm, X_test_mm)

def optimize_alpha(X_train, y_train, n_alphas=25, cv_folds=5):
    """
    Encuentra el valor óptimo de alpha para un modelo Lasso utilizando GridSearchCV.

    Args:
    - X_train: Características de entrenamiento.
    - y_train: Variable objetivo.
    - n_alphas: Número de valores de alpha para probar.
    - cv_folds: Número de pliegues en la validación cruzada.

    Return:
    - El mejor valor de alpha.
    """
    alpha_vector = np.logspace(-10, 1, n_alphas)
    param_grid = {'alpha': alpha_vector}
    grid = GridSearchCV(Lasso(), scoring='neg_mean_squared_error', param_grid=param_grid, cv=cv_folds)
    grid.fit(X_train, y_train)

    # Imprimir el mejor parámetro
    print("Best parameters: {}".format(grid.best_params_))

    # Graficar los resultados de la validación cruzada
    scores = -1 * np.array(grid.cv_results_['mean_test_score'])
    plt.semilogx(alpha_vector, scores, '-o')
    plt.xlabel('alpha', fontsize=16)
    plt.ylabel(f'{cv_folds}-Fold MSE')
    plt.ylim(min(scores) - 1, max(scores) + 1)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()

    return grid.best_params_['alpha']


def plot_lasso_paths(X_train, y_train, num_alphas=20, min_exp=-10, max_exp=0):
    """
    Plotea las trayectorias de los coeficientes de un modelo Lasso y sus normas 
    para diferentes valores de alpha.

    Args:
    - X_train: Características de entrenamiento.
    - y_train: Variable objetivo.
    - num_alphas: Número de valores de alpha para probar.
    - min_exp: Exponente mínimo para logspace de alpha.
    - max_exp: Exponente máximo para logspace de alpha.
    """
    alphas = np.logspace(min_exp, max_exp, num_alphas)
    coefs = []
    norm2_coefs = []

    for alpha in alphas:
        lasso = Lasso(alpha=alpha).fit(X_train, y_train)
        coefs.append(lasso.coef_)
        norm2_coefs.append(np.dot(lasso.coef_, lasso.coef_.T))

    plt.figure(figsize=(14, 5))

    # Coeficientes vs Alpha
    ax = plt.subplot(1, 2, 1)
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('$w_i$')
    plt.title('Coeficientes en función de la regularización')
    plt.axis('tight')

    # Norma de los coeficientes vs Alpha
    ax = plt.subplot(1, 2, 2)
    ax.plot(alphas, norm2_coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('$||\mathbf{w}||^2_2$')
    plt.title('Norma de los coeffs en función de la regularización')
    plt.axis('tight')
    plt.show()


def evaluate_model(model, X_test, y_test):
    try:
        # Realizar predicciones
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Error al predecir: {e}")
        return

    try:
        # Convertir a numpy arrays si no lo son
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
    except Exception as e:
        print(f"Error al convertir a numpy arrays: {e}")
        return
    
    try:
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
    except Exception as e:
        print(f"Error al calcular métricas: {e}")
        return
    
    try:
        # Mostrar métricas
        print(f"Resultados para {model.__class__.__name__}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    except Exception as e:
        print(f"Error al mostrar métricas: {e}")
        return
    
    try:
        # Visualizar la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.xlabel('Predicciones')
        plt.ylabel('Valores Reales')
        plt.title(f'Matriz de Confusión para {model.__class__.__name__}')
        plt.show()
    except Exception as e:
        print(f"Error al visualizar la matriz de confusión: {e}")
        return


def compute_metrics(y_true, y_pred, include_metrics=None):
    """
    Calcula métricas de rendimiento para modelos de predicción.

    Args:
    - y_true (array-like): Valores verdaderos.
    - y_pred (array-like): Valores predichos.
    - include_metrics (list, optional): Lista de métricas a incluir. Puede contener 'MAE', 'MSE', 'RMSE', 'R2'.

    Return:
    - dict: Diccionario con las métricas calculadas.
    """
    results = {}
    
    if include_metrics is None:
        include_metrics = ['MAE', 'MSE', 'RMSE', 'R2']  # Default metrics

    if 'MAE' in include_metrics:
        results['MAE'] = round(mean_absolute_error(y_true, y_pred), 3)
    if 'MSE' in include_metrics:
        results['MSE'] = round(mean_squared_error(y_true, y_pred), 3)
    if 'RMSE' in include_metrics:
        results['RMSE'] = round(root_mean_squared_error(y_true, y_pred), 3)
    if 'R2' in include_metrics:
        results['R2'] = round(r2_score(y_true, y_pred), 3)

    return results


def evaluate_model_reg(model, X_test, y_test, title='Model Evaluation', metrics_to_include=None):
    """
    Evalúa un modelo de regresión, imprime métricas y muestra un gráfico de valores reales vs. predicciones.
    
    Args:
    - model: El modelo de regresión ya entrenado.
    - X_test: Conjunto de datos de prueba (características).
    - y_test: Valores reales objetivo para el conjunto de prueba.
    - title: Título para el gráfico generado.
    - metrics_to_include: Lista de métricas a incluir en la evaluación.
    """
    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas
    metrics = compute_metrics(y_test, y_pred, include_metrics=metrics_to_include)

    # Imprimir métricas
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Graficar valores reales vs. predicciones
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Línea diagonal
    plt.xlabel('Valores Reales')
    plt.ylabel('Valores Predichos')
    plt.title(f'{title}: Predicción vs Real')
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()

    return metrics


def plot_feature_importance(model, feature_names, figsize=(10,10)):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Orden descendente
    plt.figure(figsize=figsize)
    plt.title('Importancia de las Características')
    plt.barh(range(len(indices)), importances[indices], color='cornflowerblue', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.gca().invert_yaxis()  # Invertir el eje Y para que las características más importantes aparezcan arriba
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()


def plot_feature_importances_bagging(bagging_model, feature_names, figsize=(10,12)):
    # Extraer los estimadores individuales (árboles en el caso de RandomForest)
    base_estimators = bagging_model.estimators_
    
    # Crear una matriz para guardar las importancias de cada árbol
    importances = np.array([est.feature_importances_ for est in base_estimators])
    
    # Calcular la media de las importancias a lo largo de todos los árboles
    avg_importances = np.mean(importances, axis=0)
    
    # Ordenar las importancias (y los nombres de las características correspondientes)
    indices = np.argsort(avg_importances)[::-1]
    sorted_importances = avg_importances[indices]
    sorted_features = [feature_names[i] for i in indices]
    
    # Crear la gráfica
    plt.figure(figsize=figsize)
    plt.title('Importancia de las Características')
    plt.barh(range(len(sorted_importances)), sorted_importances[::-1], color='cornflowerblue', align='center')
    plt.yticks(range(len(sorted_importances)), sorted_features[::-1])
    plt.gca().invert_yaxis()  # Invertir el eje Y para que las características más importantes aparezcan arriba
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()


def plot_confusion_matrix_with_metrics(y_test, y_pred):
    # Calcular la matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalizar para mostrar porcentajes
    
    # Calcular métricas
    precision = precision_score(y_test, y_pred, average=None, labels=np.unique(y_test))
    recall = recall_score(y_test, y_pred, average=None, labels=np.unique(y_test))
    f1 = f1_score(y_test, y_pred, average=None, labels=np.unique(y_test))

    # Crear el heatmap de la matriz de confusión
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=False, xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title('Matriz de Confusión Normalizada')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Reales')

    # Agregar las métricas al gráfico
    plt.figtext(1.0, 0.6, "Precision: " + ", ".join([f"{cls}: {p:.2f}" for cls, p in zip(np.unique(y_test), precision)]), horizontalalignment='left')
    plt.figtext(1.0, 0.5, "Recall: " + ", ".join([f"{cls}: {r:.2f}" for cls, r in zip(np.unique(y_test), recall)]), horizontalalignment='left')
    plt.figtext(1.0, 0.4, "F1 Score: " + ", ".join([f"{cls}: {f:.2f}" for cls, f in zip(np.unique(y_test), f1)]), horizontalalignment='left')

    plt.tight_layout()
    plt.show()

    # Imprimir el reporte de clasificación completo
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=np.unique(y_test)))


def plot_multiclass_roc(model, X_test, y_test, figsize=(10, 7)):
    classes = y_test.unique()
    # Binarizar las etiquetas en un one-vs-all fashion
    y_test = label_binarize(y_test, classes=classes)
    n_classes = y_test.shape[1]
    
    # Calcular las probabilidades para cada clase
    y_score = model.predict_proba(X_test)
    
    # Estructuras para guardar métricas de cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Calcular ROC y AUC para cada clase
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plotear la curva ROC para cada clase
    plt.figure(figsize=figsize)
    colors = ['blue', 'red', 'green', 'purple']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve per Class')
    plt.legend(loc="lower right")
    plt.show()


def train_and_plot_model(model, param_grid, X_train, y_train, cv=3, verbose=2):
    """
    Entrena un modelo usando GridSearchCV con los parámetros dados y plotea los resultados para un parámetro específico.
    """
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv, verbose=verbose)
    grid.fit(X_train, y_train)
    print("Best mean cross-validation score: {:.3f}".format(grid.best_score_))
    print("Best parameters: {}".format(grid.best_params_))

    # Preparar los datos para la gráfica
    results = grid.cv_results_
    scores = results['mean_test_score']
    # Asumimos que queremos plotear respecto a max_depth
    max_depth_vals = param_grid['max_depth']
    # Calcular el promedio de los scores para cada valor de max_depth
    unique_depths = np.unique([p['max_depth'] for p in results['params']])
    mean_scores = [np.mean([scores[i] for i in range(len(scores)) if results['params'][i]['max_depth'] == depth]) for depth in unique_depths]

    plt.figure(figsize=(10, 6))
    plt.plot(unique_depths, mean_scores, '-o')
    plt.title('Grid Search Scores by max_depth')
    plt.xlabel('max_depth', fontsize=16)
    plt.ylabel('10-Fold MSE', fontsize=16)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()

    return grid.best_estimator_

def train_and_plot_model_randomsearchcv(model, param_grid, X_train, y_train, cv=10, verbose=2):
    """
    Entrena un modelo usando RandomizedSearchCV con los parámetros dados y muestra los resultados para un parámetro específico.
    """

    random_search = RandomizedSearchCV(model, param_distributions=param_grid, 
                                   n_iter=20, scoring='neg_mean_squared_error', 
                                   cv=cv, verbose=verbose, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # Mejores parámetros y score
    print("Best cross-validation score (MSE):", -random_search.best_score_)
    print("Best mean cross-validation score: {:.3f}".format(random_search.best_score_))
    print("Best parameters: {}".format(random_search.best_params_))

    # Preparar los datos para la gráfica
    results = random_search.cv_results_
    scores = results['mean_test_score']
    # Calcular el promedio de los scores para cada valor de max_depth
    unique_depths = np.unique([p['max_depth'] for p in results['params']])
    mean_scores = [np.mean([scores[i] for i in range(len(scores)) if results['params'][i]['max_depth'] == depth]) for depth in unique_depths]

    plt.figure(figsize=(10, 6))
    plt.plot(unique_depths, mean_scores, '-o')
    plt.title('Grid Search Scores by max_depth')
    plt.xlabel('max_depth', fontsize=16)
    plt.ylabel('10-Fold MSE', fontsize=16)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()

    return random_search.best_estimator_


def train_evaluate_tree_model(X_train, y_train, X_test, y_test, params, cv_folds=10):
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=0), 
                               param_grid=params, 
                               cv=cv_folds,
                               scoring='accuracy',
                               return_train_score=True)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Evaluación del modelo
    train_accuracy = best_model.score(X_train, y_train)
    test_accuracy = best_model.score(X_test, y_test)
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Best Parameters:", grid_search.best_params_)
    
    # Filtrar los resultados de GridSearchCV para los mejores 'min_samples_leaf' y 'min_samples_split'
    best_leaf = grid_search.best_params_['min_samples_leaf']
    best_split = grid_search.best_params_['min_samples_split']

    mask = (np.array(grid_search.cv_results_['param_min_samples_leaf']) == best_leaf) & \
           (np.array(grid_search.cv_results_['param_min_samples_split']) == best_split)
    filtered_scores = np.array(grid_search.cv_results_['mean_test_score'])[mask]
    filtered_max_depth = [d['max_depth'] for i, d in enumerate(grid_search.cv_results_['params']) if mask[i]]

    plt.figure(figsize=(10, 5))
    plt.plot(filtered_max_depth, filtered_scores, '-o')
    plt.xlabel('max_depth')
    plt.ylabel('10-Fold Accuracy')
    plt.title('Efecto de la profundidad máxima en la puntuación de la validación cruzada de 10 pliegues')
    plt.show()
    
    return best_model


def train_random_forest(X_train, y_train, columns, param_grid, cv=3, scoring='balanced_accuracy', n_jobs=-1):
    """
    Entrena un modelo Random Forest con GridSearchCV sobre un conjunto específico de columnas y parámetros.
    
    Args:
    X_train (DataFrame): Datos de entrenamiento.
    y_train (Series): Etiquetas del conjunto de entrenamiento.
    columns (list): Lista de columnas a utilizar en el modelo.
    param_grid (dict): Grid de parámetros para la búsqueda con GridSearchCV.
    cv (int): Número de pliegues para la validación cruzada.
    scoring (str): Métrica de scoring para evaluar los modelos.
    n_jobs (int): Número de trabajos para correr en paralelo.
    """
    # Inicializar el clasificador Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=True,
                                ccp_alpha=0.0, class_weight=None, criterion='gini',
                                max_depth=None, max_features='sqrt', min_samples_leaf=1,
                                min_samples_split=2, n_jobs=n_jobs)

    # Configurar y realizar la búsqueda en malla
    grid = GridSearchCV(rf, param_grid=param_grid, cv=cv, scoring=scoring, verbose=2, n_jobs=n_jobs)
    grid.fit(X_train[columns], y_train)
    best_model = grid.best_estimator_

    # Imprimir los mejores resultados de la validación cruzada
    print("Best mean cross-validation score: {:.3f}".format(grid.best_score_))
    print("Best parameters: {}".format(grid.best_params_))

    # Extraer resultados para la visualización
    scores = np.array(grid.cv_results_['mean_test_score'])
    max_depths = param_grid['max_depth']
    mean_scores = []

    for depth in max_depths:
        indices = [i for i, params in enumerate(grid.cv_results_['params']) if params['max_depth'] == depth]
        mean_score = np.mean([scores[i] for i in indices])
        mean_scores.append(mean_score)

    # Trazar los resultados
    plt.figure(figsize=(8, 4))
    plt.plot(max_depths, mean_scores, '-o')
    plt.xlabel('max_depth')
    plt.ylabel('Balanced Accuracy')
    plt.title('Performance vs Max Depth')
    sns.despine()
    plt.show()
    
    return best_model


def train_random_forest_2(X_train, y_train, columns, param_grid, cv=3, scoring='balanced_accuracy', n_jobs=-1):
    """
    Entrena un modelo Random Forest con GridSearchCV sobre un conjunto específico de columnas y parámetros.
    
    Args:
    X_train (DataFrame): Datos de entrenamiento.
    y_train (Series): Etiquetas del conjunto de entrenamiento.
    columns (list): Lista de columnas a utilizar en el modelo.
    param_grid (dict): Grid de parámetros para la búsqueda con GridSearchCV.
    cv (int): Número de pliegues para la validación cruzada.
    scoring (str): Métrica de scoring para evaluar los modelos.
    n_jobs (int): Número de trabajos para correr en paralelo.
    """
    # Inicializar el clasificador Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=True,
                                ccp_alpha=0.0, class_weight='balanced', criterion='gini',
                                max_depth=None, max_features='sqrt', min_samples_leaf=1,
                                min_samples_split=2, n_jobs=n_jobs)

    # Configurar y realizar la búsqueda en malla
    grid = GridSearchCV(rf, param_grid=param_grid, cv=cv, scoring=scoring, verbose=2, n_jobs=n_jobs)
    grid.fit(X_train[columns], y_train)
    best_model = grid.best_estimator_

    # Imprimir los mejores resultados de la validación cruzada
    print("Best mean cross-validation score: {:.3f}".format(grid.best_score_))
    print("Best parameters: {}".format(grid.best_params_))

    # Extraer resultados para la visualización
    scores = np.array(grid.cv_results_['mean_test_score'])
    max_depths = param_grid['max_depth']
    mean_scores = []

    for depth in max_depths:
        indices = [i for i, params in enumerate(grid.cv_results_['params']) if params['max_depth'] == depth]
        mean_score = np.mean([scores[i] for i in indices])
        mean_scores.append(mean_score)

    # Trazar los resultados
    plt.figure(figsize=(8, 4))
    plt.plot(max_depths, mean_scores, '-o')
    plt.xlabel('max_depth')
    plt.ylabel('Balanced Accuracy')
    plt.title('Performance vs Max Depth')
    sns.despine()
    plt.show()
    
    return best_model


def train_random_forest_rscv(X_train, y_train):
    param_dist = {
        'max_depth': [10, 15, 20],
        'min_samples_split': [10, 20, 30],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    rf = RandomForestRegressor(random_state=0, n_estimators=200)
    random_search_rf = RandomizedSearchCV(
        rf, 
        param_dist, 
        n_iter=50, 
        cv=10, 
        scoring='neg_mean_squared_error', 
        random_state=42, 
        n_jobs=-1,  
        error_score='raise'
    )
    random_search_rf.fit(X_train, y_train)
    return random_search_rf.best_estimator_


def train_random_forest_randomized(X_train, y_train, columns, params, n_iter=100, cv=3, scoring='balanced_accuracy', n_jobs=-1, random_state=42):
    """
    Entrena un modelo Random Forest con RandomizedSearchCV sobre un conjunto específico de columnas y parámetros.
    
    Args:
    X_train (DataFrame): Datos de entrenamiento.
    y_train (Series): Etiquetas del conjunto de entrenamiento.
    columns (list): Lista de columnas a utilizar en el modelo.
    param_distributions (dict): Distribución de parámetros para la búsqueda con RandomizedSearchCV.
    n_iter (int): Número de iteraciones de configuraciones de parámetros a probar.
    cv (int): Número de pliegues para la validación cruzada.
    scoring (str): Métrica de scoring para evaluar los modelos.
    n_jobs (int): Número de trabajos para correr en paralelo.
    random_state (int): Semilla para la reproducibilidad de los resultados.
    """
    # Inicializar el clasificador Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')

    # Configurar y realizar la búsqueda aleatoria
    random_search = RandomizedSearchCV(rf, param_distributions=params, n_iter=n_iter, cv=cv, 
                                       scoring=scoring, verbose=2, random_state=random_state, n_jobs=n_jobs)
    random_search.fit(X_train[columns], y_train)
    best_model = random_search.best_estimator_

    # Imprimir los mejores resultados de la validación cruzada
    print("Best mean cross-validation score: {:.3f}".format(random_search.best_score_))
    print("Best parameters: {}".format(random_search.best_params_))

    return best_model


def train_gradient_boosting_rscv(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 150],
        'max_depth': [3, 4, 5],
        'min_samples_split': [5, 10],
        'learning_rate': [0.01, 0.025, 0.05, 0.1],
        'subsample': [0.5, 0.6, 0.7]
    }
    gbr = GradientBoostingRegressor(random_state=42)
    random_search_gbr = RandomizedSearchCV(
        gbr, 
        param_dist, 
        n_iter=100, 
        cv=10, 
        scoring='neg_mean_squared_error', 
        random_state=42, 
        n_jobs=-1
    )
    random_search_gbr.fit(X_train, y_train)
    return random_search_gbr.best_estimator_


def train_extra_trees_rscv(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': range(5, 15),
        'min_samples_split': [5, 10, 20, 30],
        'min_samples_leaf': [1, 4, 10, 15],
        'max_features': ['sqrt', 'log2']
    }
    extra_trees = ExtraTreesRegressor(random_state=42)
    random_search_et = RandomizedSearchCV(
        extra_trees, 
        param_distributions=param_dist, 
        n_iter=100, 
        scoring='neg_mean_squared_error', 
        cv=10,
        random_state=42, 
        n_jobs=-1
    )
    random_search_et.fit(X_train, y_train)
    return random_search_et.best_estimator_

# Función para entrenar y evaluar el Stacking Regressor
def train_and_evaluate_stacking(X_train, y_train, X_test, y_test, base_estimators, final_estimator, name="Model"):
    # Configuración del Stacking Regressor
    stacked_model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=10
    )
    
    # Entrenamiento y evaluación
    stacked_model.fit(X_train, y_train)
    print(f"{name} - Train R2: {stacked_model.score(X_train, y_train):.4f}, Test R2: {stacked_model.score(X_test, y_test):.4f}")