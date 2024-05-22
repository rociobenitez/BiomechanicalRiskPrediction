# Proceso de Análisis y Procesamiento de Datos

## Introducción

En esta fase del proyecto, nos enfocamos en el análisis y procesamiento de datos biomecánicos y clínicos para preparar un conjunto de datos robusto y limpio que pueda ser utilizado en el modelado predictivo de riesgo de lesiones musculoesqueléticas. Este documento justifica y detalla los pasos realizados en el cuaderno y script de preprocesamiento de los datos, siguiendo el análisis exploratorio y antes del modelado.

## 1. Configuración del Entorno de Desarrollo

Inicialmente, se preparó un entorno controlado para el desarrollo y ejecución del proyecto. Esto incluyó la configuración de un entorno virtual específico para Python, asegurando la gestión eficiente de dependencias a través de `pip`. Se instalaron bibliotecas críticas como `pandas` para manipulación de datos, `numpy` para operaciones numéricas, `scikit-learn` para algoritmos de machine learning, y `matplotlib` junto con `seaborn` para visualización de datos, soportando todas las fases de análisis y modelado.

## 2. Carga y Revisión Preliminar de Datos

Los datos fueron importados desde una **fuente de datos en formato Excel**, permitiendo una revisión inicial para entender la estructura básica y los desafíos potenciales como valores faltantes y anomalías.

## 3. Depuración de Variables Irrelevantes

Se realizó una **depuración sistemática de columnas** que no contribuían al análisis predictivo o que poseían un alto grado de redundancia. Este proceso implicó el uso de `DataFrame.drop`, optimizando el dataset para centrarse exclusivamente en variables con relevancia potencial para el análisis de lesiones.

## 4. Estandarización y Limpieza de Datos

- **Estandarización de nombres de columnas**: Se convirtieron textos a minúsculas y se eliminaron caracteres especiales o tildes para facilitar el acceso y manipulación de los datos.
- **Limpieza de datos**: Incluyó la corrección de categorías erróneas y la estandarización de entradas textuales, utilizando expresiones regulares y mapeos específicos para garantizar uniformidad y precisión.
- **Eliminación de registros no relevantes**: Se excluyeron datos de individuos menores de 15 años, basándose en consideraciones clínicas y estadísticas, utilizando filtrado condicional en `pandas`.

## 5. Imputación de Valores Faltantes

### Imputación de 'Edad' Usando Modelado Predictivo

Para abordar los valores faltantes en la variable `edad`, se desarrolló un modelo predictivo utilizando `RandomForestRegressor`, optimizado con `GridSearchCV` para garantizar la mejor imputación posible:

```python
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10]
}
model = RandomForestRegressor(random_state=42)
model_pipeline = modeling_pipeline(model, param_grid)

# Ajuste del modelo y predicción de las edades faltantes
model_pipeline.fit(X_train, y_train)
predicted_ages = best_model.predict(df_missing)
df_missing['edad'] = predicted_ages.astype(int)
df.update(df_missing)
```

### Imputación de Variables Clínicas Utilizando la Moda

Para las variables clínicas categóricas, se utilizó la imputación por moda, asegurando coherencia con las tendencias observadas en el conjunto de datos:

```python
columns_to_impute = [...]
default_values = {
    'Negativo': ['thomas psoas', 'thomas rf', ...],
    'No': [col for col in columns_to_impute if col not in ['thomas psoas', 'thomas rf', ...]]
}

for value, cols in default_values.items():
    for column in cols:
        mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else value
        df[column].fillna(mode_value, inplace=True)
```

### Imputación de Variables del Índice de Postura del Pie (FPI)

Para las variables relacionadas con el **Foot Posture Index (FPI)**, se aplicó una estrategia de imputación utilizando la moda de cada columna:

```python
df['fpi_total_i'].fillna(df['fpi_total_i'].mode()[0], inplace=True)
df['fpi_total_d'].fillna(df['fpi_total_d'].mode()[0], inplace=True)
```

## 6. Creación de Variables Derivadas y Análisis de Segmentación

### Introducción de Variables Derivadas

Se calcularon variables como el **Índice de Masa Corporal (IMC)** y la **'zona afectada'** para proporcionar una representación más clara y precisa de los factores físicos y de localización:

- **IMC**: Relaciona el peso y la altura del individuo, crucial para evaluar correlaciones potenciales con patrones de lesión.

### Racionalización de la Variable 'Zona Afectada'

Se consolidaron múltiples características en una sola variable **'zona afectada'**, agrupando categorías por áreas anatómicas generales para reducir la segmentación lateral y mejorar la generalización del modelo.

## 7. Codificación y Preparación de Variables para Modelado

### Codificación Inicial de la Variable 'Sexo'

Se aplicó una codificación simple para convertir las categorías en valores numéricos:

```python
df['sexo'] = df['sexo'].replace({'f': 0, 'm': 1})
```

### Codificación de Variables de Runscribe Marcha

Las columnas relacionadas con las métricas de marcha de Runscribe se convirtieron de formatos textuales a numéricos:

```python
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')
```

### Codificación Avanzada para Tests Clínicos

Se desarrolló una función de codificación para las variables de test clínicos, asignando un sistema de puntuación que refleja la relación entre la zona afectada y el resultado del test.

## Persistencia de Datos y Preparativos Finales

**Trabajo en Jupyter Notebook y Exportación a Script**: El procesamiento de datos se realizó inicialmente en un cuaderno de Jupyter Notebook, lo que permitió una exploración interactiva y detallada de los datos. Posteriormente, el trabajo fue estructurado y modularizado en un script Python, garantizando un código más limpio y mantenible. Finalmente, este script se exportó como un archivo `.py` ([proprocess_pipeline.py](../../../notebooks/preprocess_pipeline.py) para facilitar su ejecución en diferentes entornos.

**Almacenamiento Intermedio**: Se almacenaron los datos procesados en formato CSV tras cada fase significativa del procesamiento, garantizando la trazabilidad y la facilidad de acceso para etapas subsiguientes.

**Análisis de Correlaciones**: Se exploraron correlaciones entre las variables utilizando visualizaciones avanzadas, identificando relaciones clave que podrían influir en el poder predictivo del modelo final.

**Almacenamiento Final**: Se documentó y almacenó el conjunto de datos finalmente procesado en un archivo CSV, preparado para la fase de modelado predictivo, asegurando que todos los ajustes y transformaciones fueran preservados.

## Conclusión

El análisis y procesamiento de datos llevado a cabo en esta etapa ha permitido transformar un conjunto de datos complejo y desorganizado en una base estructurada y limpia, lista para el modelado predictivo. La combinación de técnicas avanzadas de imputación, limpieza y codificación asegura que los datos sean de alta calidad y representativos, facilitando la creación de modelos robustos y precisos para la predicción del riesgo de lesiones musculoesqueléticas.