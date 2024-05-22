#!/usr/bin/env python
# coding: utf-8

# In[3]:


import re
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, stddev, count, col, lower, trim, regexp_replace

# Configurar la variable de entorno para SPARK_LOCAL_IP
os.environ["SPARK_LOCAL_IP"] = "192.168.1.97"

# Crear una sesión de Spark
spark = SparkSession.builder \
    .appName("ETL and Metrics Calculation with Spark") \
    .config("spark.driver.bindAddress", "192.168.1.97") \
    .getOrCreate()

# Ajustar el nivel de registro a ERROR
spark.sparkContext.setLogLevel("ERROR")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar datos desde un archivo CSV
df = spark.read.csv('../data/processed/dataset_corredores.csv', header=True, inferSchema=True, sep=";")

# Crear una lista de columnas con nombres limpios
cleaned_columns = [col.strip().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace('>', '') for col in df.columns]
df = df.toDF(*cleaned_columns) 

# Estandarizar columnas de texto
columns_to_standardize = ["actividad_principal", "especialidad", "calzado_1", "calzado_2"]
for column in columns_to_standardize:
    df = df.withColumn(column, lower(trim(regexp_replace(column, r'[^\w\s]', ''))))
    
# Reemplazar comas por puntos en columnas numéricas
df = df.withColumn("marca_10k", regexp_replace("marca_10k", ",", "."))
df = df.withColumn("km/sem", regexp_replace("km/sem", ",", ".").cast("double"))

# Imputar valores faltantes en 'marca_10k' con la media
mean_value = df.select(mean('marca_10k')).collect()[0][0]
df = df.fillna({'marca_10k': mean_value})

# Calcular la moda de 'km_sem'
mode_value_row = df.groupBy("km/sem").count().orderBy("count", ascending=False).first()
mode_value = mode_value_row["km/sem"] if mode_value_row else None

# Imputar valores faltantes en 'km_sem' con la moda
if mode_value is not None:
    df = df.fillna({'km/sem': mode_value})
    
# Agrupar categorías similares en 'actividad_principal' y manejar NULL
actividad_principal_mappings = {
    r'^\s*$': 'sin_especificar',
    r'(triatln|triatln sprint|triatlon corta distancia|triatlon sprint|bombera ahora triatln sprint|triatlon  trabaja de pie y camina mucho en servicio de vehculos  ironman)': 'triatlon',
    r'(ftbol 11|ftbol 5 das sem|ftbol 8semana|ftbol 5sem y correr|futbol|portero futbol|ftbol|futbol)': 'futbol',
    r'(saltador|salto)': 'salto',
    r'(pdel|pdel y quiere hacer otro triatlón|pdel yoga gym|pdel 2 sem pilates 2 sem gym|pdel ahora parado|padel|padel gym)': 'padel',
    r'(balonmano y padel)': 'padel_balonmano',
    r'(spartan race|3000 cubierta y|10k, 5k y cross|atletismo en la granja|fredi team  3000 obstculos cross|1500 obstacules 3000 y cross|carrera(?:s)?|run(?:s)?|run and fun|atleta de lite de 1500|running amateur y hitt|obstculos cross 3000|de normal correr|y carrera(?:s)?|y carrera 10k|velocista 100m|200m|3000 y 5000|400800|1500|velocidad 100200400|mediofondo|carrera popular|carrera(?:s)?|carrera(?:s)? 10k|correr amateur|corredor amateur|10k|cross 5k 10k|running amateur|10k 5k y cross|medio fondo 800|fredi team  3000 obstculos cross)': 'correr_corta',
    r'(carrera de fondo|media maratn|de normal media maratn|maraton|media maraton|maraton de montaa y bulder|ultras de montaa|ironman|maratn y lara distancia en montaa|larga distancia|fondo antes velocidad 100200|marcha 3k 5k correr_corta y 20k)': 'correr_larga',
    r'(trailrunning|trail|maraton y trail de montaa)': 'trail',
    r'(tenista|tenis)': 'tenis',
    r'(gym|esqui gym trekking|gym y andar|rehab fuerza yoga algo de esttica|2 sem pilates 2 sem gym|gym 23 veces sem|gym y entrenamiento funcional|gym y entrenamiento personal  posturas y fuerza)': 'gym',
    r'(ciclismo|ciclismo amateur|run ciclismo|bici de montaa|bici y caminar por montaa|30kmsem ms en carretera)': 'ciclismo',
    r'(ciclismo y triatlon|ciclismo y triatln|ciclismo  y triatlon)': 'ciclismo_triatlon',
    r'(baloncesto|bsquet)': 'baloncesto',
    r'(balonmano en extremo|balonmano|balonmano profesional)': 'balonmano',
    r'(natación y carrera 10k|esqu natacin surf|rehab fuerza yoga algo de esttica y natacin|1 yoga y 1 natacin|trail artes marciales y natacin)': 'natacion',
    r'(duatlon y du cross)': 'duatlon',
    r'(1 yoga)': 'yoga',
    r'(maratn|maratn trail|maratn de montaa|maratn y lara distancia en montaa)': 'maraton',
    r'(bailarn|profesora de baile|bailarn profesional)': 'bailar',
    r'(marchadora)': 'marcha',
    r'\b(u|0|actualmente no corre|adidas glide|NULL)\b': 'sin_especificar',
    r'(chicung con ngel ventura)': 'chikung',
    r'(y esqu de montaa y)': 'esqui montaña',
    r'(oposiciones de bombero|opositora)': 'oposicion',
    r'(caminar|caminar y btt|caminar ahora con dolor limitado|caminar cada da 7km|senderismo|4 das de caminata|caminar montaa|trabaja de pie y camina mucho en servicio de vehculos|caminar daro en plano|andar y algo de bici m|andar y algo de bici m|caminar montaa|caminar ahora con dolor limitado|andar y algo de bici m|andar y algo de bici m|caminar 1h diaria|caminar daro en plano)': 'caminar',
    r'\b(y entrenamiento personal  posturas y fuerza|gym y andar|gym y entrenamiento personal  posturas y fuerza|gym y entrenamiento personal  posturas y fuerza|gym y entrenamiento personal  posturas y fuerza|gym y entrenamiento personal  posturas y fuerza|gym y entrenamiento personal  posturas y fuerza|gym y entrenamiento personal  posturas y fuerza)\b': 'gym',
    r'\b(correr_corta 5k y cross|natacin correr_corta correr_corta|marcha 3k 5k correr_corta y 20k|correr_corta 5k y cross|correr_corta 5k y cross|correr_corta 5k y cross|correr_corta 5k y cross|correr_corta 5k y cross|correr_corta 5k y cross|correr_corta 5k y cross)\b': 'correr_corta',
    r'\b(fondo antes velocidad 100200|fondo antes velocidad 100200|fondo antes velocidad 100200|fondo antes velocidad 100200|fondo antes velocidad 100200)\b': 'correr_larga',
    r'(popular| ahora con dolor limitado|profesional|2030km|hitt|montaa|amateur|sprint|3 veces sem|de fondo amateur|2 sem|1h diaria|daro en plano|ahora parado|cada da 7km|23 veces sem|profesional|and fun|y patinaje menos intensidad)': '',
    r'(y quiere hacer otro triatlon|maraton de montaa y bulder|ahora centrado en maraton)': 'correr_larga',
    r'(correr_cortaning amateur|correr_corta correr_corta|20400)': 'correr_corta',
    r'(trailcorrer_cortaning)': 'trail correr_corta',
    r'(y trail de montaa)': 'trail',
    r'(y btt)': 'ciclismo',
    r'(natacin)': 'natacion',
    r'(padel_balonmano)': 'padel balonmano',
    r'(ciclismo_triatlon)': 'ciclismo triatlon',
    r'\b(y)\b': '',
    r'(maraton de   bulder)': 'correr_larga',
    r'(200400|correr_cortaning )': 'correr_corta',
    r'(trail de)': 'trail',
    r'( pilates yoga)': 'yoga',
    r'(ciclismo oposicion)': 'ciclismo',
    r'(ciclismo triatlon)': 'triatlon',
    r'(basket 2sem)': 'baloncesto',
    r'(correr_larga trail|maraton trail|maraton)': 'correr_larga',
    r'(correr_cortaning|atletismo)': 'correr_corta',
    r'(trail triatlon)': 'triatlon',
    r'(trekking)': 'caminar',
    r'( crossfit)': 'crossfit',
    r'( natacion)': 'natacion',
    r'(padel gym)': 'padel',
    r'(remo esqui montaña|trail correr_corta|correr_larga  trail)': 'trail',
    r'(ciclismo gym|caminar ciclismo)': 'ciclismo',
    r'(golf surf)': '',
    r'(chikung)': 'chikung'
}

especialidad_mappings = {
    r'(amateur|amteur|amateur|recreacional)': 'amateur',
    r'(elite|1500 elite|nacional|2 nacional)': 'elite',
    r'(trail|y trail|ultra|ultra trail|trail cross|traillarga distancia)': 'trail',
    r'(fondo)': 'larga',
    r'(sprint|400800|8001500|400 y 800|1500|30001500|5000 1329|velocidad|5k20k)': 'corta',
    r'(triple salto y longitud|salto altura|salto longitud)': 'salto',
    r'(obstculos)': 'obstaculos',
    r'(2  1)': 'media maraton',
    r'\b(0)\b|sub40': 'sin_especificar',
    r'(trail trail|traillarga distancia|trail cross|media trail)': 'trail',
    r'(europeo)': 'elite',
    r'(media maraton)': 'larga'
}

# Aplicar las transformaciones de 'actividad_principal'
for pattern, replacement in actividad_principal_mappings.items():
    df = df.withColumn("actividad_principal", regexp_replace("actividad_principal", pattern, replacement))

df = df.withColumn("actividad_principal", trim(df["actividad_principal"]))
df = df.fillna({'actividad_principal': 'sin_especificar'})

# Aplicar las transformaciones de 'especialidad'
for pattern, replacement in especialidad_mappings.items():
    df = df.withColumn("especialidad", regexp_replace("especialidad", pattern, replacement))

df = df.withColumn("especialidad", trim(df["especialidad"]))
df = df.fillna({'especialidad': 'sin_especificar'})

# Renombrar valores NULL en las columnas de calzado
df = df.fillna({'calzado_1': 'sin_especificar', 'calzado_2': 'sin_especificar'})

# Revisar y eliminar valores nulos y duplicados
df = df.dropna().dropDuplicates()

# Lista de columnas a evaluar
columns_to_evaluate = ['pace_run', 'velocidad_run', 'step_rate_run', 'stride_length_run', 
                       'flight_ratio_run', 'power_run', 'shock_run', 'impact_gs_run', 
                       'braking_gs_run', 'footstrike_type_run', 'pronation_excursion_run', 
                       'max_pronation_velocity_run', 'peak_vertical_grf_run', 'contact_ratio_run', 
                       'stride_angle_run', 'leg_spring_stiffness_run', 'vertical_spring_stiffness_run', 
                       'vertical_grf_rate_run', 'total_force_rate_run', 'step_length_run', 
                       'pronation_excursion_mp_to_run', 'stance_excursion_fs_mp_run', 
                       'stance_excursion_mp_to_run', 'vertical_oscillation_run']

# Función para identificar outliers
def identify_outliers(df, column):
    stats = df.select(mean(col(column)).alias('mean'), stddev(col(column)).alias('stddev')).collect()
    mean_val = stats[0]['mean']
    stddev_val = stats[0]['stddev']
    
    # Definir los límites para los outliers
    lower_bound = mean_val - 3 * stddev_val
    upper_bound = mean_val + 3 * stddev_val
    
    # Filtrar outliers
    outliers = df.filter((col(column) < lower_bound) | (col(column) > upper_bound))
    return outliers

# Evaluar outliers para cada columna y guardar resultados
outliers_data = []
for column_name in columns_to_evaluate:
    outliers_df = identify_outliers(df, column_name)
    outliers_count = outliers_df.count()
    outliers_data.append((column_name, outliers_count))
    logger.info(f"Número de outliers en {column_name}: {outliers_count}")

# Guardar el DataFrame procesado en CSV
df_processed = df.toPandas()
df_processed.to_csv("../data/processed/spark/dataset_corredores_processed.csv", sep=";", index=False)

# Guardar los datos de outliers en CSV
outliers_df = pd.DataFrame(outliers_data, columns=['column', 'outliers_count'])
outliers_df.to_csv("../data/processed/spark/outliers_corredores.csv", sep=";", index=False)

# Calcular y guardar estadísticas descriptivas
selected_cols = ['edad','sexo','altura','peso','num_calzado','articulacion','localizacion','lado',
                 'actividad_principal','pace_run', 'velocidad_run', 'step_rate_run', 'stride_length_run',
                 'flight_ratio_run', 'power_run', 'shock_run', 'impact_gs_run','braking_gs_run',
                 'footstrike_type_run','pronation_excursion_run','max_pronation_velocity_run',
                 'peak_vertical_grf_run', 'contact_ratio_run','stride_angle_run','leg_spring_stiffness_run',
                 'vertical_spring_stiffness_run','vertical_grf_rate_run','total_force_rate_run',
                 'step_length_run','pronation_excursion_mp_to_run','stance_excursion_fs_mp_run',
                 'stance_excursion_mp_to_run','vertical_oscillation_run']

statistics = df[selected_cols].describe().toPandas()
statistics.to_csv("../data/processed/spark/statistics_corredores.csv", sep=";", index=False)

# Calcular y guardar matriz de correlación
correlation_matrix = df.select(columns_to_evaluate).toPandas().corr()
correlation_matrix.to_csv("../data/processed/spark/matrixcorr_corredores.csv", sep=";")

spark.stop() # Finalizar la sesión de Spark

