# Propuesta de Proyecto: Sistema Predictivo de Riesgo de Lesiones Musculoesqueléticas

## Resumen ✍🏼

El proyecto propuesto tiene como objetivo desarrollar un **sistema de análisis predictivo que evalúe el riesgo de lesión** utilizando técnicas avanzadas de minería de datos y aprendizaje automático. Aprovechando un conjunto de datos integral que abarca información biomecánica y clínica, el sistema servirá como una **herramienta valiosa para los profesionales de la salud**, ayudando en la prevención y gestión de lesiones relacionadas con el deporte.


## Declaración del Problema 📑

Las lesiones musculoesqueléticas son una preocupación prevalente en los deportes y actividades físicas, a menudo conllevando tiempos de recuperación prolongados y afectando el rendimiento y bienestar de los atletas. La capacidad de predecir y prevenir tales lesiones a través de insights basados en datos sigue siendo un desafío significativo en la medicina deportiva.

Se realiza un análisis minucioso de la documentación proporcionada para entender los requisitos, las limitaciones y las expectativas del proyecto.


### Objetivos SMART del Proyecto 🎯

1. **Específico:** 
   - Desarrollar un modelo de predicción de riesgo de lesiones musculoesqueléticas utilizando datos biomecánicos y clínicos de aproximadamente 900 pacientes.
   - Implementar y comparar al menos tres algoritmos de machine learning diferentes (regresión y clasificación) para el modelo de predicción.
   - Crear una aplicación desplegada en Google Cloud que sirva de endpoint para la ingesta y procesamiento de datos, así como la visualización de resultados.

2. **Medible:**
   - Alcanzar una precisión mínima del 70% en la predicción del riesgo de lesiones con el modelo de machine learning.
   - Identificar las 10 variables más significativas en relación con la lesión a través de un análisis exploratorio de datos (EDA) en las primeras dos semanas del proyecto.
   - Desarrollar dashboards interactivos en Tableau y reportes en d3.js para la visualización de datos.

3. **Alcanzable:**
   - Completar el análisis exploratorio de datos en las primeras dos semanas del proyecto para informar la construcción del modelo.
   - Implementar y comparar al menos tres algoritmos de machine learning diferentes en las primeras cuatro semanas del proyecto.
   - Desplegar la aplicación en Google Cloud Platform (GCP) y conectar con servicios cloud como Cloud Storage, Cloud SQL, y Cloud Functions.

4. **Relevante:**
   - Asegurar que el modelo sea interpretable clínicamente para su uso por profesionales en biomecánica.
   - Presentar los resultados de manera comprensible para usuarios no técnicos al final del proyecto.
   - Utilizar la aplicación para mejorar la prevención y gestión de lesiones musculoesqueléticas.

5. **Temporal:**
   - Presentar un prototipo funcional del sistema y un informe de resultados antes del final de las seis semanas del período del proyecto.
   - Finalizar el despliegue de la aplicación en GCP en la cuarta semana.
   - Completar la creación de visualizaciones y dashboards en Tableau y d3.js para la quinta semana.


## Fuentes de Datos 📊

El proyecto utilizará un extenso conjunto de datos proporcionado por un **equipo de investigación especializado en biomecánica** de la [Universidad San Jorge de Zaragoza](https://www.usj.es/investigacion/grupos-investigacion/unloc), que incluye **datos biomecánicos** extraídos de [Runscribe](https://runscribe.com/), junto con **historiales clínicos completos y patologías segmentadas**.


## Arquitectura del Sistema

Decidiré las herramientas y tecnologías que usaré en base a la naturaleza del dataset y los objetivos del proyecto. La arquitectura del sistema se diseñará para ser robusta, escalable y adecuada para el procesamiento de datos y la construcción de modelos predictivos.

🔗 [Configuración del Entorno de Desarrollo](./2-Development_Environment_Setup.md).


## Metodología 📝

El proyecto seguirá un **enfoque iterativo y basado en evidencia**, comenzando con el **preprocesamiento de datos** para asegurar la calidad de los mismos, seguido por un **análisis exploratorio** para descubrir patrones y correlaciones. El desarrollo principal involucrará la construcción de **modelos de aprendizaje automático para predecir el riesgo de lesiones**, con validación y pruebas para asegurar la confiabilidad del modelo.

Adoptaré una **metodología ágil** personalizada para gestionar el proyecto. Esto incluirá la creación de un **backlog personal** de tareas y metas divididas en **sprints** de una o dos semanas, asegurando un progreso constante y la capacidad de adaptarme a cambios o aprendizajes durante el desarrollo del proyecto.

🔗 [Documento del Backlog del Proyecto](3-Project_Backlog.md)


## Autogestión y Consulta 💬

Aunque trabajaré de forma independiente en el proyecto, me aseguraré de mantener un canal de comunicación abierto con el **experto en biomecánica** para resolver cualquier duda relacionada con los datos. Mis **conocimientos en biomecánica clínica** me permitirán entender y aplicar los datos de manera efectiva, y la retroalimentación del experto será esencial para validar mis interpretaciones y decisiones.


## Resultados Esperados 👩🏼‍💻

La finalización exitosa de este proyecto resultará en un sistema desplegable capaz de proporcionar evaluaciones de riesgo. Demostrará la aplicación práctica del aprendizaje automático en la atención clínica y ofrecerá una plataforma para futuras investigaciones y desarrollo.

Al finalizar este proyecto, se entregará un **modelo predictivo** junto con un dashboard interactivo y un informe técnico que documentará los hallazgos y la eficacia del modelo.


## Cronograma 📆

Se espera que el proyecto tenga una **duración de seis semanas**, con sprints iterativos siguiendo la metodología Agile [SCRUM](https://scrumguides.org/scrum-guide.html).

🔗 Enlace adicional: [Scrum (desarrollo de software)](https://es.wikipedia.org/wiki/Scrum_(desarrollo_de_software))


## Recursos 💻

Para asegurar la eficiencia del proyecto, se elaborará un presupuesto detallado que abarcará todos los recursos computacionales necesarios. Aprovecharemos la potencia y flexibilidad de **Google Cloud Platform** para alojar y procesar nuestros datos, desplegar aplicaciones y garantizar un rendimiento óptimo en cada etapa del desarrollo.


## Impacto 🏃🏼

Este proyecto tiene el potencial de contribuir en la manera en que los profesionales de la salud abordan la prevención y el tratamiento de lesiones musculoesqueléticas. Al integrar tecnologías avanzadas y análisis predictivo, se aspira a mejorar el campo hacia una práctica más personalizada y proactiva.