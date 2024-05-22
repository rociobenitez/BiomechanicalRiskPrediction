import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind, chi2_contingency


def load_data(ruta_archivo):
    """Carga los datos desde un archivo CSV."""
    df = pd.read_csv(ruta_archivo, sep=';')
    print(f"Datos cargados con dimensiones iniciales: {df.shape}")
    return df


def missing_data_analysis(df):
    # Calcular la cantidad de datos faltantes por columna
    missing_counts = df.isnull().sum()
    # Contar cuántas columnas tienen la misma cantidad de datos faltantes
    unique_counts = missing_counts.value_counts()
    return unique_counts


def set_standard_style(ax):
    """
    Aplica un estilo personalizado estándar a un objeto Axes de matplotlib.

    Args:
        ax (matplotlib.axes.Axes): El objeto Axes al que se aplicará el estilo personalizado.
    """
    # Eliminar los bordes derecho y superior
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Establecer color gris a los ejes x e y
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')

    # Establecer el tamaño de letra de las etiquetas de los ejes y los títulos
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    ax.title.set_size(12)

    # Cambiar el tamaño de letra de los ticks de los ejes
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # Establecer color de fondo y estilo de grid
    ax.set_facecolor('white')
    ax.grid(False)


def calculate_zscore(df, column):
    df[f'zscore_{column}'] = stats.zscore(df[column].copy())
    outliers = df[abs(df[f'zscore_{column}']) > 3]
    return df, outliers


def perform_anova(df, dependent_var, factor):
    model = ols(f'{dependent_var} ~ C({factor})', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table


def perform_ttest(df, dependent_var, factor):
    groups = df[factor].unique()
    if len(groups) == 2:
        group1 = df[df[factor] == groups[0]][dependent_var]
        group2 = df[df[factor] == groups[1]][dependent_var]
        t_stat, p_value = ttest_ind(group1, group2)
        return t_stat, p_value
    else:
        raise ValueError("La prueba T de Student solo es válida para dos grupos.")
    

def contingency_table(df, col1, col2):
    table = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(table)
    return table, chi2, p


def calculate_iqr_outliers(df, column):
    """Calcula outliers usando IQR"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def calculate_missing_percentage(df, selected_columns=None, visualize=False, figsize=(12, 8)):
    """
    Calcula y muestra el porcentaje de datos faltantes por columna en un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos.
        selected_columns (list, opcional): Lista de columnas a analizar. Si no se especifica, se analizan todas las columnas.
        visualize (bool, opcional): Si es True, muestra un gráfico de barras del porcentaje de datos faltantes. El valor predeterminado es False.

    Returns:
        pd.Series: Serie con el porcentaje de datos faltantes por columna, ordenada de mayor a menor.
    """
    if selected_columns is not None:
        df = df[selected_columns]

    # Calcular el porcentaje de datos faltantes por columna
    missing_percentage = df.isnull().mean() * 100

    # Ordenar las columnas por el porcentaje de datos faltantes
    missing_percentage = missing_percentage.sort_values(ascending=False)

    # Mostrar el porcentaje de datos faltantes
    print("Porcentaje de datos faltantes por columna:")
    for col, perc in missing_percentage.items():
        if perc > 0:
            print(f'{col}: {perc:.2f}%')

    # Visualización opcional
    if visualize:
        plt.figure(figsize=figsize)
        missing_percentage.plot(kind='bar', color='steelblue')
        plt.title('Porcentaje de Datos Faltantes por Columna')
        plt.xlabel('')
        plt.ylabel('Porcentaje de Datos Faltantes (%)')
        plt.xticks(rotation=90)
        sns.despine(top=True, right=True, left=False, bottom=False)
        plt.show()

    return missing_percentage


def plot_data_histogram(data, bins=20, figsize=(20, 24),
                        color='lightsteelblue', column_wrap=4):
    """
    Genera un histograma para cada columna numérica en el DataFrame.

    Args:
        data (DataFrame): El DataFrame que contiene los datos.
        bins (int, opcional): El número de contenedores para el histograma. Por defecto es 20.
        figsize (tuple, opcional): Tamaño de la figura del histograma. Por defecto es (24, 24).
        color (str, opcional): Color de los histogramas. Por defecto es 'lightsteelblue'.
        column_wrap (int): Número de histogramas por fila para evitar la superposición.
    """
    # Asegurarse de que todas las columnas se puedan convertir a numéricas, de lo contrario se convierten en NaN
    data = data.apply(pd.to_numeric, errors='coerce')

    # Calcula el número de filas necesario para la cantidad de columnas y el column_wrap especificado
    num_columns = len(data.columns)
    num_rows = int(np.ceil(num_columns / column_wrap))

    # Crea una figura con subplots en una cuadrícula de num_rows x column_wrap
    fig, axes = plt.subplots(num_rows, column_wrap,
                             figsize=figsize, constrained_layout=True)

    # Aplanar el array de ejes para facilitar su uso en un bucle
    axes = axes.flatten()

    # Generar histogramas para cada columna
    for i, (colname, coldata) in enumerate(data.items()):
        ax = axes[i]
        coldata.hist(bins=bins, ax=ax, color=color)
        ax.set_title(colname)
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frecuencia')
        set_standard_style(ax)
        # Ajustar las etiquetas para evitar la superposición
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
        for label in ax.get_yticklabels():
            label.set_rotation(45)

    # Eliminar los ejes vacíos si los hay
    for i in range(num_columns, len(axes)):
        fig.delaxes(axes[i])

    plt.show()


def plot_histogram(data, column, bins=30,
                   title="Histograma",
                   xlabel="Valor", ylabel="Frecuencia"):
    """
    Genera un histograma para una columna especificada.

    Args:
        data (DataFrame): El DataFrame que contiene los datos.
        column (str): El nombre de la columna para la cual se trazará el histograma.
        bins (int, opcional): El número de contenedores para el histograma. El valor predeterminado es 30.
        title (str, opcional): El título del gráfico. El valor predeterminado es "Histograma". 
        xlabel (str, opcional): La etiqueta para el eje x. El valor predeterminado es "Valor".
        ylabel (str, opcional): La etiqueta para el eje y. El valor predeterminado es "Frecuencia".
    """
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data[column], bins=bins, kde=True, color='royalblue')
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    set_standard_style(ax)
    plt.show()


def plot_gender_histogram(data, gender_column='Sexo',
                          bins=20, figsize=(20, 24),
                          colors=('steelblue', 'crimson'),
                          column_wrap=4):
    """
    Genera un histograma para cada columna numérica en el DataFrame y colorea
    los datos basándose en el género.

    Args:
        data (DataFrame): El DataFrame que contiene los datos.
        gender_column (str): Nombre de la columna que contiene la información de género.
        bins (int, opcional): El número de contenedores para el histograma. Por defecto es 20.
        figsize (tuple, opcional): Tamaño de la figura del histograma. Por defecto es (20, 24).
        colors (tuple, opcional): Colores para los histogramas de hombres y mujeres.
        column_wrap (int): Número de histogramas por fila para evitar la superposición.
    """
    # Asegurarse de que todas las columnas se puedan convertir a numéricas, de lo contrario se convierten en NaN
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    num_columns = len(numeric_columns)
    num_rows = int(np.ceil(num_columns / column_wrap))

    # Crea una figura con subplots en una cuadrícula de num_rows x column_wrap
    fig, axes = plt.subplots(num_rows, column_wrap,
                             figsize=figsize, constrained_layout=True)

    # Aplanar el array de ejes para facilitar su uso en un bucle
    axes = axes.flatten()

    # Iterar a través de cada columna numérica para crear histogramas
    for i, column in enumerate(numeric_columns):
        ax = axes[i]

        # Separar los datos por género
        males = data[data[gender_column] == 'M'][column]
        females = data[data[gender_column] == 'F'][column]

        # Dibujar los histogramas para cada género
        males.hist(ax=ax, bins=bins,
                   color=colors[0], alpha=0.6, label='Hombres')
        females.hist(ax=ax, bins=bins,
                     color=colors[1], alpha=0.6, label='Mujeres')

        # Configurar el título y las etiquetas
        ax.set_title(column)
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frecuencia')
        set_standard_style(ax)
        ax.legend()

    # Eliminar los ejes vacíos si los hay
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()


def plot_category_counts(y, title, figsize=(12, 6)):
    """
    Función para crear un gráfico de barras de las frecuencias de las categorías.
    
    Args:
        y (array-like): Array de datos categóricos de los que se contará la frecuencia.
        title (str): Título del gráfico.
        figsize (tuple): Tamaño de la figura del gráfico (ancho, alto).
    """
    # Calcular los conteos únicos y sus frecuencias
    unique, counts = np.unique(y, return_counts=True)
    
    # Crear un diccionario para mejor visualización
    conteo_categorias = dict(zip(unique, counts))
    
    # Configuración del gráfico
    plt.figure(figsize=figsize)
    plt.bar(conteo_categorias.keys(), conteo_categorias.values())
    plt.title(title)
    plt.xlabel('Categorías')
    plt.ylabel('Nº de Observaciones')
    plt.xticks(list(conteo_categorias.keys()))
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()


def plot_boxplot(df, title='Distribución de las características', figsize=(14, 6)):
    """
    Función para crear un boxplot de todas las columnas de un DataFrame.
    
    Parámetros:
        df (pd.DataFrame): DataFrame de pandas que contiene los datos a graficar.
        title (str): Título opcional para la gráfica.
    """
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df)
    plt.title(title)
    plt.xticks(rotation=90, ha='right')
    set_standard_style(ax)
    plt.show()


def plot_single_boxplot(data, column, title="Boxplot",
                        xlabel="", ylabel="Valor",
                        color='cornflowerblue'):
    """
    Genera un boxplot para una sola columna del DataFrame.

    Args:
        data (DataFrame): El DataFrame que contiene los datos.
        column (str): El nombre de la columna para la cual se generará el boxplot.
        title (str, opcional): El título del gráfico. Por defecto es "Boxplot".
        xlabel (str, opcional): La etiqueta para el eje x. Por defecto es una cadena vacía.
        ylabel (str, opcional): La etiqueta para el eje y. Por defecto es "Valor".
        color (str, opcional): Color de los boxplots. Por defecto es 'cornflowerblue'.
    """
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(data=data, y=column, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    set_standard_style(ax)
    plt.show()


def plot_grid_boxplots(data, columns, cols_per_row=2,
                       figsize_per_subplot=(8, 6),
                       color='cornflowerblue'):
    """
    Genera boxplots en una cuadrícula para cada columna especificada en el DataFrame.

    Args:
        data (DataFrame): El DataFrame que contiene los datos.
        columns (list): Lista de nombres de columnas para las cuales se generarán boxplots.
        cols_per_row (int): Número de boxplots por fila en la cuadrícula.
        figsize_per_subplot (tuple): Tamaño de la figura para cada subtrama individual.
        color (str, opcional): Color de los boxplots. Por defecto es 'cornflowerblue'.
    """
    rows = (len(columns) + cols_per_row - 1) // cols_per_row
    fig, axs = plt.subplots(rows, cols_per_row, figsize=(
        figsize_per_subplot[0]*cols_per_row, figsize_per_subplot[1]*rows))

    for ax, column in zip(axs.flatten(), columns):
        sns.boxplot(data=data, y=column, ax=ax, color=color)
        ax.set_title(f'Boxplot de {column}')
        set_standard_style(ax)

    # Ocultar ejes vacíos si el número de variables no llena la última fila del grid
    for ax in axs.flatten()[len(columns):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_scatter(data, x_column, y_column,
                 title="Scatter Plot",
                 xlabel="Valor X",
                 ylabel="Valor Y"):
    """
    Genera un gráfico de dispersión para dos columnas especificadas.

    Args:
        data (DataFrame): El DataFrame que contiene los datos.
        x_column (str): El nombre de la columna para la cual se trazará el gráfico en el eje x.
        y_column (str): El nombre de la columna para la cual se trazará el gráfico en el eje y.
        title (str, opcional): El título del gráfico. El valor predeterminado es "Scatter Plot". 
        xlabel (str, opcional): La etiqueta para el eje x. El valor predeterminado es "Valor X".
        ylabel (str, opcional): La etiqueta para el eje y. El valor predeterminado es "Valor Y".
    """
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=data, x=x_column,
                         y=y_column, color='royalblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    set_standard_style(ax)
    plt.show()


def plot_scatter_with_hue(data,
                          x_column, y_column,
                          hue_column,
                          title="Scatter Plot",
                          xlabel="X Value", ylabel="Y Value",
                          palette='viridis'):
    """
    Genera un gráfico de dispersión con puntos coloreados según una columna categórica.

    Args:
        data (pandas.DataFrame): DataFrame que contiene los datos a graficar.
        x_column (str): Nombre de la columna del DataFrame que se usará para el eje x.
        y_column (str): Nombre de la columna del DataFrame que se usará para el eje y.
        hue_column (str): Nombre de la columna categórica del DataFrame para diferenciar los puntos con color.
        title (str, optional): Título del gráfico.
        xlabel (str, optional): Etiqueta para el eje x.
        ylabel (str, optional): Etiqueta para el eje y.
    """
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=data, x=x_column, y=y_column,
                         hue=hue_column, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax = set_standard_style(ax)
    plt.legend(title=hue_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_count(data, column,
               title="Count Plot",
               xlabel="Categoría", ylabel="Count",
               figsize=(8, 6), palette="bone"):
    """
    Genera un count plot para una columna especificada excluyendo valores nulos y 'null'.

    Args:
        data (DataFrame): El DataFrame que contiene los datos.
        column (str): El nombre de la columna para la cual se trazará el count plot.
        title (str, opcional): El título del gráfico. El valor predeterminado es "Count Plot". 
        xlabel (str, opcional): La etiqueta para el eje x. El valor predeterminado es "Categoría".
        ylabel (str, opcional): La etiqueta para el eje y. El valor predeterminado es "Count".
        figsize (tuple, opcional): Tamaño de la figura del gráfico. Por defecto es (8, 6).
        palette (str, opcional): Paleta del gráfico. Por defecto es 'bone'.
    """
    if column in data.columns:
        plt.figure(figsize=figsize)
        ax = sns.countplot(x=data[column].replace('null', None).dropna(),
                           palette=palette,
                           hue=data[column].replace('null', None).dropna())
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
        set_standard_style(ax)
        plt.show()
    else:
        print(f"La columna {column} no existe en el DataFrame.")


def plot_count_including_nulls(data, column,
                               title="Count Plot",
                               xlabel="Categoría", ylabel="Count",
                               figsize=(8, 6), palette="bone"):
    """
    Genera un gráfico de barras de conteo para una columna especificada,
    incluyendo una barra para valores nulos.

    Args:
        dataframe (pandas.DataFrame): El DataFrame que contiene los datos.
        column_name (str): El nombre de la columna para la cual se generará el gráfico de conteo.
        title (str, opcional): El título del gráfico. El valor predeterminado es "Count Plot". 
        xlabel (str, opcional): La etiqueta para el eje x. El valor predeterminado es "Categoría".
        ylabel (str, opcional): La etiqueta para el eje y. El valor predeterminado es "Count".
        figsize (tuple, opcional): Tamaño de la figura del gráfico. Por defecto es (8, 6).
        palette (str, opcional): Paleta de colores para el gráfico. Por defecto es 'bone'
    """
    # Crear una copia de la columna con valores nulos y vacíos reemplazados por una categoría 'Null'
    column_with_nulls_and_empty = data[column].replace(
        ' ', 'Null').fillna('Null')

    plt.figure(figsize=figsize)
    ax = sns.countplot(x=column_with_nulls_and_empty, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    set_standard_style(ax)
    plt.show()


def plot_count_horizontal(data, column,
                          title="Count Plot",
                          xlabel="Frecuencia",
                          ylabel="Categoría",
                          figsize=(12, 18),
                          palette="coolwarm"):
    """
    Genera un count plot horizontal para una columna especificada,
    excluyendo valores nulos y 'null',
    con colores que varían según la frecuencia.
    """
    if column in data.columns:
        plt.figure(figsize=figsize)
        # Calculando frecuencias
        frequencies = data[column].value_counts()
        # Ordenar por frecuencia
        ordered = frequencies.index
        # Crear el gráfico
        ax = sns.barplot(x=frequencies.values, y=ordered,
                         palette=palette, hue=frequencies.values)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        set_standard_style(ax)
        plt.show()
    else:
        print(f"La columna {column} no existe en el DataFrame.")


def plot_numerical_columns(data, column_names, num_cols=2, ylabel='', palette='coolwarm'):
    num_plots = len(column_names)
    num_rows = num_plots // num_cols + (num_plots % num_cols > 0)

    plt.figure(figsize=(num_cols * 7, num_rows * 5))

    for index, column in enumerate(column_names, 1):
        plt.subplot(num_rows, num_cols, index)
        # Asegurarse de que los datos son numéricos
        cleaned_data = pd.to_numeric(data[column], errors='coerce')
        order = sorted(cleaned_data.dropna().unique())
        ax = sns.barplot(x=cleaned_data.value_counts(
        ).index, y=cleaned_data.value_counts().values, order=order, palette=palette)
        set_standard_style(ax)
        plt.title(f'Distribución de {column}')
        plt.xlabel(column)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_superimposed_histograms(data1, data2,
                                 xlabel='Valor', ylabel='Frecuencia',
                                 labeldata1='Marcha', labeldata2='Carrera',
                                 bins=20,
                                 figsize=(20, 24),
                                 colors=('steelblue', 'darkorange'),
                                 column_wrap=4):
    """
    Genera histogramas superpuestos para cada columna numérica en dos DataFrames.

    Args:
        data1 (DataFrame): El primer DataFrame que contiene los datos.
        data2 (DataFrame): El segundo DataFrame que contiene los datos.
        xlabel (str, opcional): La etiqueta para el eje x. El valor predeterminado es "Valor".
        ylabel (str, opcional): La etiqueta para el eje y. El valor predeterminado es "Frecuencia".
        labeldata1 (str, opcional): Etiqueta del primer histograma. El valor predeterminado es "Marcha".
        labeldata2 (str, opcional): Etiqueta del segundo histograma. El valor predeterminado es "Carrera".
        bins (int, opcional): El número de contenedores para el histograma. Por defecto es 20.
        figsize (tuple, opcional): Tamaño de la figura del histograma. Por defecto es (20, 24).
        colors (tuple, opcional): Colores para los histogramas de cada DataFrame.
        column_wrap (int): Número de histogramas por fila para evitar la superposición.
    """
    # Asegurarse de que todas las columnas se puedan convertir a numéricas, de lo contrario se convierten en NaN
    data1 = data1.apply(pd.to_numeric, errors='coerce')
    data2 = data2.apply(pd.to_numeric, errors='coerce')

    # Calcula el número de filas necesario para la cantidad de columnas y el column_wrap especificado
    num_columns = len(data1.columns)
    num_rows = int(np.ceil(num_columns / column_wrap))

    # Crea una figura con subplots en una cuadrícula de num_rows x column_wrap
    fig, axes = plt.subplots(num_rows, column_wrap,
                             figsize=figsize, constrained_layout=True)

    # Aplanar el array de ejes para facilitar su uso en un bucle
    axes = axes.flatten()

    # Generar histogramas para cada columna
    for i, colname in enumerate(data1.columns):
        ax = axes[i]
        # Histograma para el primer DataFrame
        data1[colname].hist(bins=bins, ax=ax, color=colors[0],
                            alpha=0.5, label=labeldata1)
        # Histograma para el segundo DataFrame
        data2[colname].hist(bins=bins, ax=ax, color=colors[1],
                            alpha=0.5, label=labeldata2)
        ax.set_title(colname)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right')
        set_standard_style(ax)
        # Ajustar las etiquetas para evitar la superposición
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')

    # Eliminar los ejes vacíos si los hay
    for i in range(num_columns, len(axes)):
        fig.delaxes(axes[i])

    plt.show()


def catplot(data, x, hue, title, xlabel='', ylabel='', palette='tab20c'):
    """
    Genera y muestra un gráfico categórico de tipo 'countplot' utilizando Seaborn. 
    El gráfico cuenta las ocurrencias de la categoría especificada por 'hue' 
    para cada categoría única en la columna 'x'.

    Args:
        data (pandas.DataFrame): DataFrame que contiene los datos para graficar.
        x (str): Nombre de la columna del DataFrame que se utilizará para el eje x del gráfico.
        hue (str): Nombre de la columna del DataFrame cuyas categorías se usarán para colorear los segmentos del gráfico.
        title (str): Título del gráfico.
        xlabel (str, opcional): Etiqueta para el eje x. Si no se proporciona, el eje x no tendrá etiqueta.
        ylabel (str, opcional): Etiqueta para el eje y. Si no se proporciona, el eje y no tendrá etiqueta.
        palette (str, opcional): Paleta de colores a usar en el gráfico. Por defecto es 'tab20c'.
    """

    # Filtrar valores 'null' y ordenar las categorías de 'hue' alfabéticamente
    data = data.replace('null', None).dropna(subset=[x, hue])
    # Asegurarse de que hue es de tipo str para evitar problemas al ordenar
    data[hue] = data[hue].astype(str)
    data = data.sort_values(by=hue)

    cat_plot = sns.catplot(
        x=data[x].replace('null', None).dropna(),
        hue=hue,
        data=data,
        kind='count',
        height=6,
        aspect=1.5,
        palette=palette,
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    ax = cat_plot.ax
    set_standard_style(ax)
    plt.show()


def stacked_barplot(data, x, hue, title, xlabel='', ylabel='', palette='tab20c'):
    """
    Genera y muestra un gráfico de barras apiladas.

    Args:
        data (pandas.DataFrame): DataFrame con los datos a graficar.
        x (str): Nombre de la columna del DataFrame para el eje x.
        hue (str): Nombre de la columna del DataFrame para la codificación de color de las barras.
        title (str): Título del gráfico.
        xlabel (str, opcional): Etiqueta del eje x.
        ylabel (str, opcional): Etiqueta del eje y.
        palette (str, opcional): Paleta de colores para las barras.
    """
    # Agrupa los datos y calcula el conteo
    grouped = data.groupby([x, hue]).size().unstack().fillna(0)

    # Crea un gráfico de barras apiladas
    ax = grouped.plot(
        kind='bar',
        stacked=True,
        colormap=palette,
        figsize=(10, 7)
    )

    # Configuraciones de estilo
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    set_standard_style(ax)
    plt.legend(title=hue)
    plt.tight_layout()
    plt.show()


def plot_heatmap(corr_matrix, figsize=(12, 10), cmap="GnBu", title='', annot=True):
    """
    Esta función genera un mapa de calor para una matriz de correlación dada.

    Parámetros:
    corr_matrix (DataFrame): Pandas DataFrame que contiene la matriz de correlación.
    figsize (tuple): Tamaño de la figura del mapa de calor (ancho, alto).
    cmap (str): Nombre del mapa de colores a utilizar.
    title (str): Título del mapa de calor.
    """
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, mask=mask, linewidths=.1, annot=annot,
                cmap=cmap, cbar_kws={"shrink": .8}, fmt=".2f")
    plt.title(title)
    plt.xlabel('')
    plt.ylabel('')
    plt.show()


def plot_categorical_columns(data, column_names, num_cols=2, xlabel='', ylabel='Conteo', palette='coolwarm'):
    """
    Genera gráficos de conteo para columnas categóricas específicas en un DataFrame.

    Args:
        data (pd.DataFrame): El DataFrame con los datos.
        column_names (list of str): Lista de nombres de columnas para visualizar.

    Returns:
        None: Esta función no devuelve nada pero genera gráficos de barras.
    """
    num_plots = len(column_names)
    num_rows = num_plots // num_cols + (num_plots % num_cols > 0)

    plt.figure(figsize=(num_cols * 7, num_rows * 5))

    for index, column in enumerate(column_names, 1):
        ax = plt.subplot(num_rows, num_cols, index)
        categories_order = data[column].dropna().unique()
        sns.countplot(x=column, data=data,
                      order=categories_order, palette=palette)
        set_standard_style(ax)
        plt.title(f'{column}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def check_exclusivity(df, col1, col2, col3):
    """
    Verifica la exclusividad entre las columnas.

    Args:
        df (DataFrame): El DataFrame con los datos.
        col1 (str): Nombre de la columna con valores 'Sí' o 'No' para la normalidad.
        col2 (str): Nombre de la primera columna de test.
        col3 (str): Nombre de la segunda columna de test.
    """
    # Casos donde col1 es 'No' y uno de los otros es positivo
    condition = ((df[col1] == 'No') & ((df[col2] == 'Izquierdo') | (df[col2] == 'Derecho')) |
                 (df[col1] == 'No') & ((df[col3] == 'Izquierdo') | (df[col3] == 'Derecho')))

    # Resumen de la exclusividad
    summary = condition.value_counts()
    print(f"Resumen de exclusividad entre {col1}, {col2} y {col3}:")
    print(summary)


def stacked_barplot_excluding_no(data, columns, hue, title, palette='tab20c', nrows=2, ncols=2, figsize=(15, 10)):
    """
    Genera y muestra un gráfico de barras apiladas excluyendo los valores 'No', organizado en un número definido de filas y columnas.

    Args:
        data (pandas.DataFrame): DataFrame con los datos a graficar.
        columns (list): Lista de nombres de columnas para excluir los valores 'No'.
        hue (str): Nombre de la columna del DataFrame para la codificación de color de las barras.
        title (str): Título del gráfico.
        palette (str, opcional): Paleta de colores para las barras.
        nrows (int): Número de filas de subplots.
        ncols (int): Número de columnas de subplots.
        figsize (tuple): Dimensiones totales de la figura.
    """
    # Crear una figura con un grid de subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, tight_layout=True)

    # Aplanar la matriz de ejes para un bucle fácil
    axes = axes.flatten()

    for i, column in enumerate(columns):
        # Filtrar el DataFrame para excluir filas con 'No' en la columna actual
        filtered_data = data[data[column] != 'No']

        # Calcular la tabla de contingencia para contar ocurrencias
        ct = pd.crosstab(filtered_data[hue], filtered_data[column])

        # Gráfico de barras apiladas en el subplot correspondiente
        ax = ct.plot(kind='bar', stacked=True, colormap=palette, ax=axes[i])
        set_standard_style(ax)
        axes[i].set_title(f'{title}: {column}')
        axes[i].set_xlabel(hue)
        axes[i].set_ylabel('Frecuencia')
        axes[i].legend(title=column, loc='upper right')

    # Si hay más subplots que columnas de datos, eliminar los subplots extras
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    plt.show()


def plot_original_vs_imputed(df, df_validation, df_validation_imputed, columns_runscribe_walk, N=5):
    """
    Grafica la comparación de valores originales e imputados para columnas seleccionadas.

    Parámetros:
    - df (pandas.DataFrame): DataFrame con los datos originales.
    - df_validation (pandas.DataFrame): DataFrame con datos de validación que incluyen valores faltantes.
    - df_validation_imputed (pandas.DataFrame): DataFrame con datos de validación donde los valores faltantes han sido imputados.
    - columns_runscribe_walk (list of str): Lista de nombres de columnas a graficar.
    - N (int): Número de subgráficos a mostrar. Por defecto es 5.

    Esta función crea una serie de subgráficos, cada uno comparando los valores originales e imputados para una columna específica.
    Los subgráficos se organizan en una cuadrícula de 2 columnas.
    """
    # Calcular el número de filas necesarias para N subgráficos en 2 columnas.
    # Añade uno en caso de un número impar de gráficos para asegurar que todos se muestren.
    rows = (N + 1) // 2
    # Ajustar el tamaño de la figura apropiadamente.
    fig, axs = plt.subplots(rows, 2, figsize=(16, rows * 4))

    # Aplanar el array axs para simplificar la indexación en un solo bucle.
    axs = axs.flatten()

    for i, column in enumerate(columns_runscribe_walk[:N]):
        hidden_indices = df_validation[column].isna() & ~df[column].isna()

        # Gráfico de dispersión para los puntos de datos originales.
        axs[i].scatter(df.index, df[column], alpha=0.6,
                       label='Original', color='cornflowerblue')
        # Gráfico de dispersión para los puntos de datos imputados donde faltaban datos originales.
        axs[i].scatter(df_validation_imputed.index[hidden_indices], df_validation_imputed[column]
                       [hidden_indices], alpha=0.6, label='Imputado', color='crimson')

        set_standard_style(axs[i])
        axs[i].set_title(
            f'Comparación de Valores Originales y Imputados para {column}')
        axs[i].set_xlabel('Índice')
        axs[i].set_ylabel('Valores')
        axs[i].legend()

    # Manejar cualquier subgráfico extra en la última fila si N es impar.
    if N % 2 != 0:
        axs[-1].axis('off')  # Ocultar el último subgráfico si no es necesario.

    plt.tight_layout()
    plt.show()
