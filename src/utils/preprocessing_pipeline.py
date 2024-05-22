import pandas as pd
import numpy as np
import logging
from unidecode import unidecode

import sys
sys.path.append('../src/utils')
from preprocess import clean_column, clean_articulacion, clean_localizacion, clean_lado

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnPreparer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # No hay parámetros para inicializar

    def fit(self, X, y=None):
        return self  # Nada que hacer aquí

    def transform(self, X):
        try: 
            #logging.info("Primer valor (header candidato): %s", X.iloc[0].to_dict())
            new_header = X.iloc[0]
            X = X[1:]
            X.columns = new_header
            X.reset_index(drop=True, inplace=True)
            X.columns = [str(col).strip() for col in X.columns]

            if len(X.columns) > 151:
                X.columns.values[35:93] = [
                    str(col) + '_walk' for col in X.columns[35:93]]
                X.columns.values[93:151] = [
                    str(col) + '_run' for col in X.columns[93:151]]
            
            return X
        except Exception as e:
            logging.error("Error al preparar las columnas: %s", e)
            raise


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop  # Lista de columnas a eliminar

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        #logging.info("Columnas antes de eliminar: %s", X.columns)
        X = X.drop(self.columns_to_drop, axis=1)
        #logging.info("Columnas después de eliminar: %s", X.columns)
        return X


class ColumnCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, corrections=None):
        self.corrections = corrections or {
            'rotula ascencida': 'rotula ascendida',
            'no de calzado': 'num calzado'
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un pandas.DataFrame")
        
        # Normalizar los nombres de las columnas
        X.columns = [unidecode(col).lower() for col in X.columns]
        X.columns = [self.corrections.get(col, col) for col in X.columns]
        #logging.info("Columnas después de corrección de nombres: %s", X.columns)

        # Aplicar limpieza general de columnas de texto
        for col in X.select_dtypes(include='object').columns:
            X[col] = clean_column(X[col])
        logging.info("Limpieza general de contenido completada para columnas de texto.")
        logging.info("Total NaN: %d", X.isna().sum().sum())

        # Aplicar funciones de limpieza específicas
        X['articulacion'] = clean_articulacion(X['articulacion'])
        X['localizacion'] = clean_localizacion(X['localizacion'])
        X['lado'] = clean_lado(X['lado'])

        return X


class AgeFilter(BaseEstimator, TransformerMixin):
    def __init__(self, minimum_age=14):
        self.minimum_age = minimum_age

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        original_count = len(X)
        X = X[(X['edad'] > self.minimum_age) | X['edad'].isna()]
        filtered_count = len(X)
        logging.info("Registros antes de filtrar la edad: %d, después de filtrar: %d", original_count, filtered_count)
        logging.info("Total NaN: %d", X.isna().sum().sum())
        return X


class EncodeCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_encode):
        self.column_to_encode = column_to_encode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #X = X.copy()
        original_values = X[self.column_to_encode].unique()
        X[self.column_to_encode] = X[self.column_to_encode].replace({'f': 0, 'm': 1})
        updated_values = X[self.column_to_encode].unique()
        logging.info("Valores originales en '%s': %s, después de codificar: %s", self.column_to_encode, original_values, updated_values)
        logging.info("Total NaN: %d", X.isna().sum().sum())
        return X


class SpeedFilter(BaseEstimator, TransformerMixin):
    def __init__(self, min_speed=4.0, max_speed=5.5):
        self.min_speed = min_speed
        self.max_speed = max_speed

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        count_in_range = X[(X['velocidad_walk'] >= self.min_speed) & (X['velocidad_walk'] <= self.max_speed)].shape[0]
        #count_out_of_range = X.shape[0] - count_in_range
        #logging.info("Número de registros dentro del rango [%.1f, %.1f]: %d", self.min_speed, self.max_speed, count_in_range)
        #logging.info("Número de registros fuera del rango [%.1f, %.1f]: %d", self.min_speed, self.max_speed, count_out_of_range)
        X = X[(X['velocidad_walk'] >= self.min_speed) & (X['velocidad_walk'] <= self.max_speed)]
        X = X[(X['articulacion'] != 'complejo')]
        return X
    

class AgeImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = RandomForestRegressor(random_state=42)
        self.model_pipeline = None

    def fit(self, X, y=None):
        # Convertir 'edad' a numérica
        X['edad'] = pd.to_numeric(X['edad'], errors='coerce').astype('Int64')
        
        # Eliminar registros donde 'edad' es NaN para garantizar que el modelo se entrene con datos completos
        df_full = X.dropna(subset=['edad'])
        X_train = df_full[['sexo', 'altura', 'peso', 'num calzado']]
        y_train = df_full['edad']
        logging.info("Valores faltantes en 'edad' después de eliminar registros sin 'edad': %d", y_train.isna().sum())

        # Dividir datos en conjuntos de entrenamiento y prueba, estratificando por 'sexo' para mantener proporciones
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=df_full['sexo']
        )

        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10]
        }
        self.model_pipeline = GridSearchCV(
            Pipeline([('model', self.model)]),
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        self.model_pipeline.fit(X_train_split, y_train_split)

        # Mostrar los mejores parámetros y modelo
        best_params = self.model_pipeline.best_params_
        best_model = self.model_pipeline.best_estimator_
        logging.info("Mejores hiperparámetros: %s", best_params)
        logging.info("Mejor modelo para imputar la edad: %s", best_model)

        best_model.fit(X_train_split, y_train_split)  # Entrenar el modelo
        return self

    def transform(self, X):
        df_missing = X[X['edad'].isnull()][['sexo', 'altura', 'peso', 'num calzado']]
        predicted_ages = self.model_pipeline.predict(df_missing)
        df_missing['edad'] = predicted_ages.astype(int)
        X.update(df_missing)
        return X
    

class ClinicalDataImputer(BaseEstimator, TransformerMixin):
    def __init__(self, default_values):
        self.default_values = default_values

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for value, cols in self.default_values.items():
            for column in cols:
                mode_value = X[column].mode().iloc[0] if not X[column].mode().empty else value
                X[column].fillna(mode_value, inplace=True)
                #logging.info("Valores nulos en columna '%s' después de imputar con '%s': %d", column, mode_value, X[column].isna().sum())
        
        logging.info("Imputación completada. Total NaN: %d", X.isna().sum().sum())
        return X


class IMCCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_convert):
        self.columns_to_convert = columns_to_convert

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logging.info("Nº de columnas numéricas antes de la codificación: %d", X.select_dtypes(include=['number']).columns.size)
        
        for col in self.columns_to_convert:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        logging.info("Número de columnas numéricas después de la codificación: %d", X.select_dtypes(include=['number']).columns.size)
        
        imc = (X['peso'] / (X['altura'] / 100) ** 2).round(2)
        X = pd.concat([X, imc.rename('imc')], axis=1)
        
        #logging.info("Estadísticas descriptivas de la columna 'imc':\n%s", X['imc'].describe())
        logging.info("Nº de valores nulos en la columna 'imc': %d", X['imc'].isna().sum())
        
        return X


class ReplaceAndVerify(BaseEstimator, TransformerMixin):
    def __init__(self, column, replacements):
        self.column = column
        self.replacements = replacements

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.column] = X[self.column].replace(self.replacements)
        return X


class CreateAffectedZoneColumn(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def normalize_column(text):
            if pd.isna(text):
                return text
            return unidecode(text.lower())

        for column in ['articulacion', 'localizacion', 'lado']:
            X[column] = X[column].apply(normalize_column)
        
        X['zona afectada'] = X['articulacion'] + '_' + X['localizacion'] + '_' + X['lado']
        logging.info("Nº de valores únicos en 'zona afectada': %d", X['zona afectada'].nunique())
        return X


class MapAffectedZones(BaseEstimator, TransformerMixin):
    def __init__(self, group_mapping):
        self.group_mapping = group_mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['zona afectada'] = X['zona afectada'].map(self.group_mapping)
        unmapped = X[X['zona afectada'].isnull()]
        if not unmapped.empty:
            logging.warning("Hay categorías sin mapear: %s", unmapped['zona afectada'].unique())
        else:
            logging.info("Todas las categorías han sido mapeadas correctamente.")
        
        return X


class ClinicalTestCoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns_tests):
        self.columns_tests = columns_tests

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.columns_tests:
            X[col] = X.apply(lambda row: self.codificar_tests(row, col), axis=1)
        return X

    def codificar_tests(self, row, column):
        zona = row['zona afectada']
        valor_test = row[column]

        if valor_test == 'Izquierda':
            valor_test = 'Izquierdo'
        elif valor_test == 'Derecha':
            valor_test = 'Derecho'
        
        if zona == 'sin patologia':
            return 0

        if valor_test in ['No', 'Negativo']:
            return 0
        
        if not any(zona.endswith(suffix) for suffix in ['_i', '_d', '_b']):
            zona += '_b'

        if valor_test == 'Bilateral':
            return 3

        if (zona.endswith('_i') and valor_test == 'Izquierdo') or (zona.endswith('_d') and valor_test == 'Derecho'):
            return 1

        if (zona.endswith('_i') and valor_test == 'Derecho') or (zona.endswith('_d') and valor_test == 'Izquierdo'):
            return 2

        if zona.endswith('_b'):
            if valor_test in ['Izquierdo', 'Derecho']:
                return 1

        return 0


class ConvertColumnsToNumeric(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logging.info("Nº de columnas numéricas antes de la codificación: %d", X.select_dtypes(include=['number']).columns.size)
        for col in self.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        logging.info("Número de columnas numéricas después de la codificación: %d", X.select_dtypes(include=['number']).columns.size)
        return X


class AdjustContactRatio(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['contact ratio_walk'] = X['contact ratio_walk'] / 2
        return X


class ImputeRunscribeData(BaseEstimator, TransformerMixin):
    def __init__(self, cols_runscribe_walk, random_state=None):
        self.cols_runscribe_walk = cols_runscribe_walk
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            np.random.seed(self.random_state)
            hide_percentage = 0.1  # 10% de los datos
            df_validation = X.copy()

            # Ocultamos un porcentaje de los datos de cada columna
            for column in self.cols_runscribe_walk:
                values_to_hide = int(len(df_validation[column]) * hide_percentage) 

                # Seleccionar índices aleatorios para ocultar
                indices_to_hide = np.random.choice(df_validation[column].dropna().index, values_to_hide, replace=False)
                df_validation.loc[indices_to_hide, column] = np.nan  # 'Ocultar' los valores asignando NaN

            logging.info("Total NaN after hiding values: %s", df_validation.isna().sum().sum())

            # Crear un pipeline de imputación y escalado como antes
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', KNNImputer(n_neighbors=5))
            ])

            # Realizar la imputación en el conjunto de validación
            df_validation_imputed_scaled = pipeline.fit_transform(df_validation[self.cols_runscribe_walk])
            df_validation_imputed = pd.DataFrame(
                df_validation_imputed_scaled,
                columns=self.cols_runscribe_walk,
                index=df_validation.index
            )
            df_validation_imputed = pd.DataFrame(
                pipeline.named_steps['scaler'].inverse_transform(df_validation_imputed_scaled),
                columns=self.cols_runscribe_walk,
                index=df_validation.index
            )

            logging.info("Imputación y escalado de datos de Runscribe completada.")

            # Calcular métricas de validación
            for column in self.cols_runscribe_walk:
                # Índices que fueron 'ocultados' artificialmente
                hidden_indices = df_validation[column].isna() & ~X[column].isna()

                # Valores reales y valores imputados
                true_values = X[column][hidden_indices]
                imputed_values = df_validation_imputed[column][hidden_indices]

                # Calcular y mostrar el MAE y RMSE
                mae = mean_absolute_error(true_values, imputed_values)
                rmse = np.sqrt(mean_squared_error(true_values, imputed_values))

                logging.info('%s - MAE: %f, RMSE: %f', column, mae, rmse)

            # Reemplazar las columnas imputadas en el DataFrame original
            X[self.cols_runscribe_walk] = df_validation_imputed[self.cols_runscribe_walk]
            logging.info("Reemplazo de columnas imputadas completado.")
            logging.info("Total NaN after replacing columns: %s", X.isna().sum().sum())
            
            # Verificar las filas con NaN restantes y registrar detalles
            nan_rows = X[X.isna().any(axis=1)]
            if not nan_rows.empty:
                logging.info("Filas con NaN después de la imputación:\n%s", nan_rows)
            else:
                logging.info("No hay filas con NaN después de la imputación.")

            return X

        except Exception as e:
            logging.error("Error during runscribe data imputation: %s", e)
            return None


class FinalDataPreparation(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop_1, columns_to_drop_2, save_path_1, save_path_2, save_path_3):
        self.columns_to_drop_1 = columns_to_drop_1
        self.columns_to_drop_2 = columns_to_drop_2
        self.save_path_1 = save_path_1
        self.save_path_2 = save_path_2
        self.save_path_3 = save_path_3

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            # Eliminar las primeras columnas
            logging.info("Eliminando columnas: %s", self.columns_to_drop_1)
            X.drop(self.columns_to_drop_1, axis=1, inplace=True)
            logging.info("Shape after first column drop: %s", X.shape)

            # Guardar el DataFrame en un archivo CSV antes de eliminar las columnas 'articulacion', 'localizacion' y 'lado'
            logging.info("Guardando el DataFrame en: %s", self.save_path_1)
            X.to_csv(self.save_path_1, sep=";", index=False, encoding='utf-8')

            # Eliminar las segundas columnas
            logging.info("Eliminando columnas: %s", self.columns_to_drop_2)
            X.drop(self.columns_to_drop_2, axis=1, inplace=True)
            logging.info("Shape after second column drop: %s", X.shape)

            # Guardar el DataFrame en un archivo CSV
            logging.info("Guardando el DataFrame en: %s", self.save_path_2)
            X.to_csv(self.save_path_2, sep=";", index=False, encoding='utf-8')

            # Eliminar columnas adicionales para el archivo final
            final_columns_to_drop = [
                'zona afectada', 'num calzado', 'm1 hipermovil', 'thomas psoas', 'thomas rf', 'thomas tfl',
                'ober', 'arco aplanado', 'arco elevado', 'm1 dfx', 'm5 hipermovil', 'arco transverso disminuido',
                'm1 pfx', 'arco transverso aumentado', 'hlf', 'hl', 'hr', 'hav', 'index minus', 'tfi', 'tfe',
                'tti', 'tte', 'ober friccion', 'popliteo', 't_hintermann', 'jack normal', 'jack no reconstruye',
                'pronacion no disponible', '2heel raise', 'heel raise', 'fpi_total_i', 'fpi_total_d',
                'tibia vara proximal', 'tibia vara distal', 'rotula divergente', 'rotula convergente',
                'rotula ascendida', 'genu valgo', 'genu varo', 'genu recurvatum', 'genu flexum', 'lunge'
            ]
            logging.info("Eliminando columnas finales: %s", final_columns_to_drop)
            X.drop(final_columns_to_drop, axis=1, inplace=True)
            logging.info("Dimensiones del DataFrame después de la eliminación de columnas: %s", X.shape)

            # Guardar el DataFrame en un archivo CSV final
            logging.info("Guardando el DataFrame final en: %s", self.save_path_3)
            X.to_csv(self.save_path_3, sep=";", index=False, encoding='utf-8')

            return X
        except Exception as e:
            logging.error("Error during final data preparation: %s", e)
            return None


# Pipeline para procesamiento
def preprocessing_pipeline(columns_to_drop,
                           default_values,
                           columns_to_convert,
                           replacements,
                           group_mapping,
                           columns_tests):
    steps = [
        ('preparer_columns', ColumnPreparer()),
        ('drop_columns', ColumnDropper(columns_to_drop=columns_to_drop)),
        ('clean_columns', ColumnCleaner({
            'rotula ascencida': 'rotula ascendida',
            'no de calzado': 'num calzado'
        })),
        ('filter_age', AgeFilter(minimum_age=14)),
        ('filter_speed', SpeedFilter(min_speed=4.0, max_speed=5.5)),
        ('encode_categorical', EncodeCategorical(column_to_encode='sexo')),
        ('impute_ages', AgeImputer()),
        ('impute_clinical_data', ClinicalDataImputer(default_values=default_values)),
        ('calculate_imc', IMCCalculator(columns_to_convert=columns_to_convert)),
        ('replace_and_verify', ReplaceAndVerify(column='lado', replacements=replacements)),
        ('create_affected_zone', CreateAffectedZoneColumn()),
        ('map_affected_zones', MapAffectedZones(group_mapping=group_mapping)),
        ('convert_columns_to_numeric', ConvertColumnsToNumeric(columns=columns_to_convert)),
        ('adjust_contact_ratio', AdjustContactRatio()),
        ('clinical_test_coder', ClinicalTestCoder(columns_tests=columns_tests)),
    ]
    return Pipeline(steps)


def preprocessing_pipeline_part2(cols_runscribe_walk):
    steps = [
        ('impute_runscribe_data', ImputeRunscribeData(cols_runscribe_walk=cols_runscribe_walk, random_state=42)),
        ('final_data_preparation', FinalDataPreparation(
            columns_to_drop_1=[
                'stride angle_walk', 'leg spring stiffness_walk', 'vertical spring stiffness_walk',
                'max pronation velocity_walk', 'vertical grf rate_walk', 'flight ratio_walk', 'peak vertical grf_walk',
                'pnca ap varo', 'pnca ap valgo', 'pnca rp varo', 'pnca rp valgo', 'power_walk'
            ],
            columns_to_drop_2=['articulacion', 'localizacion', 'lado'],
            save_path_1='../data/processed/dataset_complete_segmented.csv',
            save_path_2='../data/processed/dataset_complete.csv',
            save_path_3='../data/processed/dataset_marcha.csv'
        ))
    ]
    return Pipeline(steps)