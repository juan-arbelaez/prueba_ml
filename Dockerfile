# Usar una imagen ligera de Python 3.9
FROM python:3.9-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Evitar caché en la instalación de dependencias
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias para Jupyter y NLP
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar los archivos de dependencias
COPY requirements.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Instalar Jupyter Notebook y otras utilidades
RUN pip install --no-cache-dir jupyter notebook

# Copiar el código del proyecto al contenedor
COPY . .

# Exponer el puerto 8888 para acceder a Jupyter Notebook
EXPOSE 8888

# Definir el comando por defecto para iniciar Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
