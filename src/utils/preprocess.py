import pandas as pd
import numpy as np
from unidecode import unidecode


# Función para cargar datos
def cargar_datos(filepath):
    df = pd.read_excel(filepath)
    print(f"Datos cargados con dimensiones iniciales: {df.shape}")
    # Limpiar espacios en blanco de los nombres de columnas.
    df.columns = [col.strip() for col in df.columns]
    return df


# Función para renombrar y limpiar columnas
def preparar_columnas(df):
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    df.reset_index(drop=True, inplace=True)

    # Eliminar espacios extra en nombres de columnas.
    df.columns = [col.strip() for col in df.columns]
    df.columns.values[35:93] = [
        str(col) + '_walk' for col in df.columns[35:93]]
    df.columns.values[93:151] = [
        str(col) + '_run' for col in df.columns[93:151]]
    print("Columnas renombradas para diferenciar entre marcha y carrera.")
    return df

# Función para eliminar columnas irrelevantes
def eliminar_columnas_irrelevantes(df, columns_to_drop):
    df.drop(columns_to_drop, axis=1, inplace=True)
    print(f"Datos después de eliminar columnas: {df.shape}")
    return df


def clean_column(column):
    """
    Elimina espacios en blanco al principio y final, y transforma a minúsculas,
    solo si la columna es de tipo string. Los valores nulos se mantienen como NaN para un tratamiento específico posterior.

    Args:
        column (pandas.Series): La serie a procesar.

    Returns:
        pandas.Series: La serie procesada, conservando los valores nulos originales.
    """
    # Comprobar si los elementos de la columna son de tipo string
    if pd.api.types.is_string_dtype(column):
        # Limpiar strings: eliminar espacios y convertir a minúsculas
        column = column.str.strip().str.lower()
        # Reemplazar cadenas vacías con NaN, pero mantener NaN existentes
        column = column.replace('', np.nan)
    return column


def clean_articulacion(column):
    """
    Limpia y unifica las categorías de la columna 'Articulación'.

    Args:
        column (pandas.Series): Serie que representa la columna 'Articulación'.

    Returns:
        pandas.Series: Serie con las categorías unificadas y limpias.
    """
    # Mapeo de valores originales a las nuevas categorías
    replacements = {
        r'\brodilla\b': 'rodilla',
        r'\brodila\b': 'rodilla',  # Asume que es un error tipográfico común
        r'\bespalda\b': 'espalda',
        r'\blumbar\b': 'espalda',
        r'\bpierna\b': 'pierna',
        r'\bpie\b': 'pie',
        r'\bpies\b': 'pie',  # Unificar plural
        r'\btobillo\b': 'tobillo',
        r'\bcadera\b': 'cadera',
        r'\bmuslo\b': 'muslo',
        r'\bcomplejo\b': 'complejo',
        r'\bsin afectacion\b': 'sin afectación'
    }

    # Convertir a minúsculas y aplicar reemplazos
    cleaned_column = column.str.lower().str.strip()
    for pattern, replacement in replacements.items():
        cleaned_column = cleaned_column.str.replace(
            pattern, replacement, regex=True)

    # Reemplazar cualquier término que no coincida con los especificados por 'No especificado'
    cleaned_column = cleaned_column.where(
        cleaned_column.isin(replacements.values()), 'No especificado')

    return cleaned_column


def clean_localizacion(column):
    """
    Limpia y unifica las categorías de la columna 'localización', asignando 'No especificado' a cualquier
    valor que no coincida con las categorías definidas o que sea nulo, vacío o erróneo.

    Args:
        column (pandas.Series): Serie que representa la columna 'localización'.

    Returns:
        pandas.Series: Serie con las categorías unificadas y limpias.
    """
    # Crear un mapa de reemplazo para las variantes detectadas
    replacements = {
        'amterior': 'anterior',
        'anterior - posterior': 'anteroposterior',
        'medial-lateral': 'mediolateral',
        'medail': 'medial',
        'interna': 'medial',
        'planta proximal': 'plantar proximal',
        'posteriro': 'posterior',
        'posterior ': 'posterior',
        'lateral ': 'lateral',
        'planta distal': 'plantar distal'
    }

    # Lista de valores válidos después de los reemplazos
    valid_values = ['anterior', 'posterior', 'medial', 'lateral', 'distal', 'proximal',
                    'anteroposterior', 'anteromedial', 'anterolateral', 'lumbar',
                    'mediolateral', 'posteromedial', 'posterolateral', 'plantar',
                    'plantar proximal', 'plantar distal', 'dorsal distal', 'dorsal proximal']

    # Convertir todo a minúsculas y reemplazar según el mapa
    column = column.str.lower().str.strip()
    column = column.replace(replacements)

    # Asignar 'No especificado' a valores no reconocidos
    column = column.apply(
        lambda x: x if x in valid_values else 'no especificado')

    # Manejar valores nulos
    column = column.fillna('no especificado')

    return column


def clean_lado(column):
    """
    Limpia y unifica las categorías de la columna 'Lado' según las reglas definidas.
    Los valores se normalizan a 'left', 'right', 'bilateral', 'bilateral + left', 
    'bilateral + right', o 'null'.

    Args:
        column (pandas.Series): Serie de pandas que representa la columna 'Lado'.

    Returns:
        pandas.Series: Serie con las categorías unificadas y limpias.
    """
    # Mapeo de valores originales a las nuevas categorías
    map_to_left = ['i', 'izquierdo', 'izquierda',
                   'pie izquierdo', 'left', 'izdo', 'izqueirdo']
    map_to_right = ['d', 'derecho', 'derecha',
                    'pie derecho', 'dereho', 'derehca']
    map_to_bilateral = ['b', 'bilateral', 'bilatera', 'empezó con el derecho y ahora bilateral.',
                        'derecho e izquierdo.', 'bilaterales', 'bilateral.', 'pie bilateral']
    map_to_bilateral_left = ['más lado izquierdo', 'bilateral (+en i)', 'bilateral más izquierdo', 'bilateral + izquierda',
                             'bilateral (+izquierda)', 'bilateral +izq.', 'bilateral, más izquierdo', 'bilateral (+izquierdo)']
    map_to_bilateral_right = ['bilateral (+pd)', 'bilateral +pd', 'biltaeral (+d)', 'bilateral +dx', 'bilateral (+derecho)', 'bilateral más en derecha', 'bilateral, ahora más derecho',
                              'bilateral, peor derecho', 'bilateral (+d)', 'derecho (alguna ligera izquierda)', 'derecho/bilateral', 'bilateral más derecha', 'mas derecho']
    map_to_null = ['pi + larga?', 'talón', '-', np.nan, 'cadera', 'null']

    # Reemplazar valores según el mapeo
    new_column = column.str.strip().str.lower()
    new_column = new_column.replace(map_to_left, 'i')
    new_column = new_column.replace(map_to_right, 'd')
    new_column = new_column.replace(map_to_bilateral, 'b')
    new_column = new_column.replace(map_to_bilateral_left, 'b + i')
    new_column = new_column.replace(
        map_to_bilateral_right, 'b + r')
    new_column = new_column.replace(map_to_null, 'no especificado')

    return new_column


def filter_age(df, age_threshold=14):
    """
    Filtra el DataFrame para excluir a individuos menores de un umbral de edad especificado
    mientras mantiene los registros con 'edad' no especificada (NaN).

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        age_threshold (int): Umbral de edad mínima para los registros que se mantendrán.

    Returns:
        pd.DataFrame: DataFrame filtrado.
    """
    print(f"Valores faltantes en 'edad' antes de filtrar: {df['edad'].isna().sum()}")

    original_count = len(df)
    df_filtered = df[(df['edad'] > age_threshold) | df['edad'].isna()]
    filtered_count = len(df_filtered)

    print(f"Nº original de registros: {original_count} \nNº de registros después de eliminar menores de {age_threshold} años: {filtered_count}")

    min_age = df_filtered['edad'].min()
    print(f"Edad mínima en el DataFrame filtrado: {min_age if pd.notna(min_age) else 'No disponible'}")

    nan_count = df_filtered['edad'].isna().sum()
    print(f"Valores faltantes en la columna 'edad' después del filtro: {nan_count}")
    print(f"Total NaN en el DataFrame después del filtro: {df_filtered.isna().sum().sum()}")

    return df_filtered


def impute_clinical_data(df, default_values):
    """
    Imputa los valores faltantes de las columnas clínicas especificadas usando la moda o un valor predeterminado.
    
    Args:
    df (pd.DataFrame): DataFrame que contiene los datos.
    columns_to_impute (list): Lista de columnas para imputar.
    default_values (dict): Diccionario que asigna valores predeterminados a grupos de columnas.
    
    Returns:
    pd.DataFrame: DataFrame con valores imputados.
    """
    for value, cols in default_values.items():
        for column in cols:
            mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else value
            df[column].fillna(mode_value, inplace=True)
    
    print("Imputación completada. Total NaN:", df.isna().sum().sum())
    return df


def replace_and_verify(df, column, replacements):
    # Reemplazar valores según el diccionario de reemplazos
    df[column] = df[column].replace(replacements)
    
    # Verificación de los cambios realizados
    print(f"Valores únicos en '{column}': {df[column].unique()}")
    return df


def normalize_column(text):
    if pd.isna(text):
        return text
    return unidecode(text.lower())


def create_affected_zone_column(df):
    # Normalizar columnas relevantes
    for column in ['articulacion', 'localizacion', 'lado']:
        df[column] = df[column].apply(normalize_column)
    
    # Crear la columna 'zona afectada'
    df_temp = pd.concat([df, df['articulacion'] + '_' + df['localizacion'] + '_' + df['lado']], axis=1)
    df_temp.columns = df.columns.tolist() + ['zona afectada']
    df = df_temp
    
    print(f"Nº de valores únicos en 'zona afectada': {df['zona afectada'].nunique()}")
    return df


def map_affected_zones(df, group_mapping):
    # Aplicar mapeo a la columna 'zona afectada'
    df['zona afectada'] = df['zona afectada'].map(group_mapping)
    
    # Verificar si hay alguna categoría que no se haya mapeado correctamente
    unmapped = df[df['zona afectada'].isnull()]
    if not unmapped.empty:
        print("Hay categorías sin mapear:")
        print(unmapped['zona afectada'].unique())
    else:
        print("Todas las categorías han sido mapeadas correctamente.")
    
    return df


# Función de codificación
def codificar_tests(row, column):
    zona = row['zona afectada']
    valor_test = row[column]

    # Normalizar valores de "Izquierda/Izquierdo" y "Derecha/Derecho"
    if valor_test == 'Izquierda':
        valor_test = 'Izquierdo'
    elif valor_test == 'Derecha':
        valor_test = 'Derecho'
        
    # Caso para "sin patología"
    if zona == 'sin patologia':
        return 0

    # Mapear "No" y "Negativo" a un concepto unificado
    if valor_test in ['No', 'Negativo']:
        return 0
    
    # Chequear si la zona no tiene sufijo específico y no es 'sin patología'
    if not any(zona.endswith(suffix) for suffix in ['_i', '_d', '_b']):
        zona += '_b'  # Tratar como bilateral si no hay sufijo

    # Mapear "Bilateral" tanto en test como en zona afectada
    if valor_test == 'Bilateral':
        return 3  # Ambos lados afectados

    # Chequear si el test coincide con el lado afectado
    if (zona.endswith('_i') and valor_test == 'Izquierdo') or (zona.endswith('_d') and valor_test == 'Derecho'):
        return 1

    # Chequear si el test es contralateral al lado afectado
    if (zona.endswith('_i') and valor_test == 'Derecho') or (zona.endswith('_d') and valor_test == 'Izquierdo'):
        return 2

    # Casos especiales para bilateral
    if zona.endswith('_b'):
        if valor_test in ['Izquierdo', 'Derecho']:
            return 1  # Mismo miembro afectado (considerando que 'bilateral' incluye ambos lados)

    # Si no hay coincidencias específicas, retornar cero
    return 0