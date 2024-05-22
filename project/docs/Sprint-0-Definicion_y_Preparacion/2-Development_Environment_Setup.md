# Configuración del Entorno de Desarrollo

Antes de iniciar cualquier tarea o sección del proyecto, se configuró adecuadamente el entorno de desarrollo. Este paso es importante para garantizar la coherencia y eficacia en los análisis y el modelado predictivo. A continuación, se detallan las **herramientas y el proceso seguido** para poner en marcha dicho entorno.


## Herramientas y Software 🛠️

- **Python 3.12**: Lenguaje de programación de alto nivel con extenso soporte en librerías especializadas para ciencia de datos y aprendizaje automático.
- **Jupyter Notebook**: Interfaz interactiva que facilita la creación y compartición de documentos que contienen código en vivo, ecuaciones, visualizaciones y texto narrativo.
- **Librerías de Python**:
  - `Pandas`: Para la manipulación y análisis de datos.
  - `NumPy`: Soporte para grandes matrices y arrays multidimensionales.
  - `Matplotlib` y `Seaborn`: Herramientas de visualización de datos.
  - `Scikit-learn`: Simple y eficiente herramienta para análisis predictivo de datos.
  - `TensorFlow`: Librería de aprendizaje automático para flujo de trabajo de alto rendimiento.
- **Git y GitHub**: Control de versiones y repositorio para el almacenamiento y colaboración de código.

## Proceso de Configuración 🔧

### Paso 1: Instalación de Python y Bibliotecas

Primero, instalamos Python 3.12 y las bibliotecas necesarias utilizando `pip` o `conda`. Se recomienda usar la última versión compatible de cada biblioteca para evitar conflictos.

🔗 [Descargas de Python](https://www.python.org/downloads/)

### Paso 2: Creación de un Entorno Virtual

Para mantener las cosas ordenadas, creamos un entorno virtual aislado con `venv` o `conda env`. Esto nos permitirá gestionar paquetes sin interferir con otros proyectos o la configuración del sistema.

### Paso 3: Verificación del Entorno

Después, ejecutamos un script de verificación para confirmar que todo esté bien instalado y configurado. Este script realizará pruebas básicas importando cada biblioteca y ejecutando comandos simples para validar la configuración.

### Configuración Adicional para PySpark 🛠️

Para la configuración específica de PySpark, que incluye la instalación de Java y la configuración de Spark, revisa el documento [Configuración del Entorno PySpark](/Etapas/Sprint-1-Arquitectura_y_Validacion_de_datos/PySpark_Environment_Setup.md).


## Instrucciones de Uso 👩🏼‍💻

### Activación del Entorno Virtual

Antes de comenzar cualquier sesión de trabajo, es importante activar el entorno virtual para asegurarte de que estás utilizando las versiones correctas de las herramientas y bibliotecas:

```bash
source nombre_entorno/bin/activate
```

### Inicio de Jupyter Notebook

Una vez activado el entorno virtual, inicia Jupyter Notebook para acceder a un entorno interactivo de desarrollo:

```bash
jupyter notebook
```

## Buenas Prácticas ✅

- **Documentación**: Llevar un registro de cualquier cambio en la configuración o en las dependencias del proyecto.
- **Actualizaciones Regulares**: Mantener todas las herramientas y bibliotecas actualizadas para aprovechar las mejoras y correcciones de seguridad.
- **Backup del Entorno**: Considerar exportar la lista de dependencias con `pip freeze` o `conda list --export` para tener un respaldo del entorno.

Con este entorno de desarrollo preparado y verificado, el proyecto está listo para avanzar en la experimentación y desarrollo de modelos predictivos de forma eficiente y controlada. Este documento ofrece una guía clara y técnica para configurar y usar el entorno de desarrollo, asegurando que cualquier persona que se una al proyecto pueda hacerlo sin problemas.