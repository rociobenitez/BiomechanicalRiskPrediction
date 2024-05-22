# Utilizar una imagen base con Python y Java preinstalado
FROM openjdk:11-jre-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de requisitos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

# Instalar Spark
RUN wget -qO- https://archive.apache.org/dist/spark/spark-3.0.1/spark-3.0.1-bin-hadoop2.7.tgz | tar xvz -C /opt/
ENV SPARK_HOME=/opt/spark-3.0.1-bin-hadoop2.7
ENV PATH=$SPARK_HOME/bin:$PATH

# Exponer el puerto 5000 para Flask
EXPOSE 5000

# Comando para iniciar la aplicación
CMD ["python", "app.py"]