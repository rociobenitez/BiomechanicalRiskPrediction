# Análisis Exploratorio de Datos (EDA)

El análisis exploratorio de datos (EDA) nos ha permitido comprender mejor el dataset de datos antropométricos, biomecánicos de marcha y carrera de RunScribe, así como los resultados de test clínicos, e identificar patrones, relaciones y anomalías.

**Enlaces a los archivos**:

📑 [EDA Inicial en Jupyter Notebook](../../../notebooks/EDA_inicial.ipynb)
📑 [EDA Post-Procesamiento Datos en Jupyter Notebook](../../../notebooks/EDA-post-procesamiento.ipynb)

## Estructura General del Notebook

### 1. Preparación de los Datos
- **Configuración del Entorno**: Importación de librerías necesarias y configuración inicial.
- **Carga de Datos desde archivo .xlsx**: Uso de `pandas` para cargar y explorar los datos.
- **Adaptación de los Nombres de Columnas**: Ajuste de nombres para asegurar consistencia y claridad.

### 2. Inspección Inicial de los Datos
- **Comprobación de Tipos de Datos**: Verificación y ajuste de los tipos de datos.
- **Identificación de Columnas con Mayor Número de Datos Faltantes**: Evaluación del porcentaje de datos faltantes por columna.
- **Verificación de Filas Duplicadas**: Identificación y tratamiento de registros duplicados.

### 3. Análisis Exploratorio de Datos (EDA)
- **Diferenciación de Nombres de Columnas para Datos de Runscribe**: Separación de métricas de marcha y carrera.
- **Agrupación Provisional de Columnas por Categoría**:
    - Demográficas y de perfil
    - Datos relacionados con patología
    - IB_Report
    - Runscribe_walk
    - Runscribe_run
    - IB_Report 2
- **Análisis de Datos Faltantes**: 
    - **Métrica**: Porcentaje de datos faltantes por columna.
    - Estrategias de tratamiento de datos faltantes.
- **Estadísticas Descriptivas Básicas**: 
    - **Métricas**: Media, mediana, moda, varianza, desviación estándar, percentiles.
- **Visualización de Datos**: 
    - Uso de gráficos como histogramas, boxplots, scatter plots y pair plots para visualizar la distribución y relaciones entre las variables.
- **Análisis de Correlación**: 
    - Cálculo de la matriz de correlación para identificar relaciones lineales entre las variables.

### 4. Análisis por Grupos
#### Grupo 1 - Perfil / Demográficos
- Conversión de tipo de datos
- Distribución y análisis de outliers
- Análisis de registros con códigos repetidos
- Medidas de Asociación - **Métrica**: Coeficiente de correlación
- Análisis de la varianza - **Métricas**: ANOVA y Prueba T de Student

#### Grupo 2 - Patología / Zona Afectada
- Revisión de valores faltantes
- Conversión de tipo de datos
- Identificación de valores únicos por columna
- Unificación de categorías con funciones personalizadas
- Análisis de la distribución de zonas afectadas
- **Métrica**: Tabla de Contingencia

#### Grupo 3 - IB_Report / Historia Clínica
- Detección de alta incidencia de datos faltantes

#### Grupo 4 - Runscribe_walk / Datos Biomecánicos de Marcha
- Revisión de valores faltantes
- Eliminación de espacios en los nombres de las columnas
- Conversión a valores numéricos
- Evaluación de la eliminación o no de registros
- Medidas de Asociación - **Métrica**: Coeficiente de correlación
- Visualización con histogramas
- Análisis de la distribución y detección de outliers (boxplots)

#### Grupo 5 - Runscribe_run / Datos Biomecánicos de Carrera
- Revisión de valores faltantes
- Eliminación de espacios en los nombres de las columnas
- Conversión a valores numéricos
- Revisión de análisis estadístico
- Análisis de la distribución y detección de outliers (boxplots)
- Medidas de Asociación - **Métrica**: Coeficiente de correlación.
- Visualización con histogramas

#### Grupo 6 - IB_Report 2 / Tests Clínicos
- Revisión de valores faltantes
- Análisis de valores únicos por columna
- Medidas de Asociación - **Métrica**: Coeficiente de correlación


### Conclusiones

A lo largo de esta etapa del análisis exploratorio de datos, he abordado varias tareas clave que me han permitido comprender mejor el dataset y preparar el terreno para los análisis y modelos predictivos posteriores.

- **Renombrado de Columnas de RunScribe**: He diferenciado entre las columnas de datos de marcha y de carrera para evitar confusiones y asegurar una mayor claridad en el análisis.

- **Eliminación de Columnas con Muchos Valores Faltantes**: Identifiqué las columnas con una alta proporción de datos faltantes, con el objetivo de mejorar la calidad y consistencia del dataset.

- **Estandarización del Procedimiento de Recolección de Datos**: Observé una notable disparidad en los datos recolectados. Este hallazgo subraya la necesidad de establecer procesos de recolección de datos más estandarizados para futuras investigaciones.

- **Exclusión de Datos de Historia Clínica**: Decidí excluir los datos relacionados con la historia clínica debido a la alta incidencia de datos faltantes y la heterogeneidad de la información. Esta decisión me permitirá centrarme en datos más completos y consistentes.

- **Selección de Columnas Relevantes**: Identifiqué las columnas más importantes para el análisis posterior, priorizando aquellas con menor cantidad de valores faltantes y mayor relevancia clínica.

- **Distribución del Sexo**: Noté una mayor representación del sexo masculino en el dataset. Este desequilibrio se tendrá en cuenta en los análisis posteriores para evitar sesgos.

- **Eliminación de Registros con Edades Inferiores a 15 Años**: Eliminaré los registros con edades inferiores a 15 años para centrar el análisis en datos relevantes y reducir la heterogeneidad.

- **Estrategias de Imputación y Codificación de Variables**: Desarrollé estrategias para la imputación de datos faltantes y la codificación de variables categóricas, preparando el dataset para los modelos predictivos.

- **Creación de Nuevas Columnas (IMC y Zona Afectada)**: Se añadirán columnas derivadas como el índice de masa corporal (IMC) y una columna consolidada de zona afectada, enriqueciendo el dataset con información adicional útil.

- **Análisis de Outliers en Datos Biomecánicos**: Realicé un análisis exhaustivo de outliers en los datos biomecánicos. Detecté varios puntos extremos que podrían ser variaciones individuales extremas o errores en los sensores. Estos puntos serán investigados más a fondo para asegurar su validez.

- **Foco en Datos de Marcha**: Decidí centrarme exclusivamente en los datos de marcha, excluyendo los de carrera para simplificar el análisis y debido a la menor cantidad de valores faltantes en estos datos.

- **Distribución de Patologías por Articulación y Localización**: Analicé la distribución de patologías, identificando que el pie y la rodilla son las zonas más afectadas. También observé diferencias notables entre géneros en la frecuencia de problemas articulares.

- **Segmentación Futura de la Población**: Para futuros análisis, consideré la posibilidad de segmentar la población entre corredores y no corredores, lo que podría proporcionar información más específica y relevante.



sprint|400800|8001500|400 y 800|1500|30001500|5000 1329|velocidad