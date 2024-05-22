# Sistema Predictivo de Riesgo de Lesiones Musculoesqueléticas

## Introducción

Este proyecto representa el desarrollo principal del **bootcamp de Big Data y Machine Learning**, aplicando todos los conocimientos adquiridos. El objetivo es crear una **aplicación desplegada en el ecosistema de Google Cloud**, demostrando un entendimiento de la arquitectura en la nube. Además, se implementarán **dos modelos predictivos: uno de regresión y otro de clasificación**, utilizando datos reales (entorno a 900 sujetos) proporcionados por un [equipo de investigación](https://www.usj.es/investigacion/grupos-investigacion/unloc). El proyecto también abarca el **procesamiento y visualización** de datos para ofrecer una solución integral en la predicción del riesgo de lesiones musculoesqueléticas.


## Cronograma del Proyecto

**Inicio del Proyecto:** 10 de abril
**Entrega Final:** 23 de mayo


## Índice de Etapas del Proyecto

1. **[Sprint 0: Definición y Preparación](./Sprint-0-Definicion_y_Preparacion/)** - *(10 de abril - 15 de abril)*
    - **Objetivos** Definir objetivos SMART y alcance, planificación detallada
    - **Tareas**: Configuración del entorno de desarrollo y creación del backlog. El dataset de origen unifica diferentes fuentes: datos demográficos de historiales clínicos, datos biomecánicos de RunScribe (marcha y carrera) y datos de formularios de tests clínicos. Los datos de optogait y baropodometría se reservaron para análisis posteriores.
    - **Entregables**:
        - [Documento de Alcance del Proyecto](./Sprint-0-Definicion_y_Preparacion/1-Project_Proposal.md)
        - [Entorno de Desarrollo Configurado](./Sprint-0-Definicion_y_Preparacion/2-Development_Environment_Setup.md)
        - [Backlog del Proyecto](./Sprint-0-Definicion_y_Preparacion/3-Project_Backlog.md)

2. **[Sprint 1: Análisis Exploratorio y ETL y cálculo de métricas con Spark](./Sprint-1-EDA_y_ETL/)** - *(22 de abril - 28 de abril)*
    - **Objetivos**: Validar y entender los datos proporcionados.
    - **Tareas**: Análisis exploratorio de datos (EDA), identificar anomalías y outliers, selección de características, plantear la limpieza y preprocesamiento de datos.
    - **Entregables:**
      - [Reporte detallado de EDA](./Sprint-1-EDA_y_ETL/EDA.md)
      - Proceso seguido en el EDA en un cuaderno de Jupyter [.ipynb](../../notebooks/EDA_inicial.ipynb) y [.html](../../notebooks/EDA_inicial.html)
      - [Documento explicativo del Proyecto ETL y Análisis de Métricas con Spark](./Sprint-1-EDA_y_ETL/ETL-Spark.md)
      - [Configuración inicial del Entorno de Desarrollo para PySpark](./Sprint-1-EDA_y_ETL/PySpark_Environment_Setup.md)

3. **[Sprint 2: Limpieza y Preprocesamiento de Datos. Modelado con Machine Learning](./Sprint-2-Modelado/)** - *(29 de abril - 8 de mayo)*
    - **Objetivos**: Limpiar y preprocesar el conjunto de datos, construir modelos predictivos iniciales.
    - **Tareas:** Normalización, estandarización y transformación de datos, imputación de valores faltantes y codificación de variables categóricas, selección de características, entrenamiento de modelos, evaluación preliminar, validación cruzada y ajuste de hiperparámetros.
    - **Entregables:**
      - Dataset limpio y preprocesado
      - Scripts o notebooks de preprocesamiento (ubicados en la carpeta [/notebooks](../../notebooks/))

4. **[Sprint 3: Arquitectura, Almacenamiento e Ingesta](./Sprint-3-Arquitectura/)** - *(9 de mayo - 16 de mayo)*
    - **Objetivos**: Definir y desplegar la arquitectura en Google Cloud Platform (GCP)
    - **Tareas:** Crear una aplicación Flask para la ingesta de datos, conectar la aplicación con servicios cloud (Cloud Storage, Cloud SQL, Cloud Functions), preparar archivos para el despliegue de la aplicación, realizar pruebas en producción.
    - **Entregables:**
        - [Diagrama de la arquitectura planteada](./Sprint-3-Arquitectura/img/arquitectura-google-cloud.jpg)
        - [Documento explicativo de la arquitectura](./Sprint-3-Arquitectura/Architecture.md)
        - [Documento de configuración realizada en GCP](./Sprint-3-Arquitectura/Configuracion_Stack_GCP.md)
        - [Aplicación deplegada en GCP](https://sistemas-predictivo-lesiones.ew.r.appspot.com/clasificacion)

5. **[Sprint 4: Visualización](./Sprint-4-Visualizacion/)** - *(17 de mayo - 19 de mayo)*
    - **Objetivos**: Crear un dashboard interactivo realizado en Tableau
    - **Tareas**: Crear un dashboard con Tableau, desarrollar un reporte con d3.js en la aplicación Flask
    - **Entregables:**
      - [Documento explicativo del trabajo realizado en Tableau](./Sprint-4-Visualizacion/tableau.md)
      - [Documento explicativo del report en D3.js](./Sprint-4-Visualizacion/d3-js.md)
      - Página en la app de Flask con los gráficos planteados


6. **[Sprint 5: Optimización, Documentación y Presentación de Resultados](./Sprint-5-Optimización_y_Presentación/)** - *(19 de mayo - 23 de mayo)*
    - **Objetivos**: Documentar y presentar el proyecto.
    - **Tareas:** Ordenar y seleccionar archivos relevantes del proyecto, documentar completamente el proyecto, revisar los entregables, preparar la presentación final.
    - **Entregables**:
        - Documentación completa del proyecto
        - Enlace al repositorio de GitHub con todos los archivos del proyecto
        - Presentación final del proyecto


## Conclusión

Este README proporciona una guía de las etapas del proyecto, siguiendo una metodología ágil para asegurar un desarrollo organizado y eficiente. Cada etapa está documentada en archivos individuales para una mayor profundización, permitiendo a cualquier evaluador entender claramente los pasos seguidos y los resultados obtenidos en el proyecto.