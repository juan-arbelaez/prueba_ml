# Análisis de Sentimiento con Machine Learning y CI/CD

#### Este proyecto implementa un pipeline de Machine Learning para análisis de sentimientos utilizando modelos como RandomForest, SVM y XGBoost. Incluye un flujo automatizado de CI/CD/CT con GitHub Actions, asegurando calidad del código, pruebas automatizadas y despliegue continuo.

Comentarios previos:

Quería entregar el proyecto con una imagen Docker por el tema de los tamaños de los archivos que estaba manejando se me empezó a complicar un poco la configuración y más aun siendo en local por terminos de RAM y CPU, tuve que hacer muchas configuraciones para poder ejecutarlo pero no quise correr el riesgo de que esas configuraciones no surtieran efecto correctamente en la persona que fuera a revisar esta prueba. Por tal motivo, deje el Dockerfile como lo habría configurado inicialmente para poder ejecutar tanto scripts de Python como Jupyter Notebook para hacer el análisis inicial, pero mi recomendación final es generar un entorno virtual con virtualenv basado en Python 3.9 e instalar el archivo requirements con ```pip install -r requirements.txt``` (aunque para temas de dependencias, yo sugiero utilizar uv por encima de pip, pero por temas de estandarizaciones lo hago asi para esta prueba).

#### IMPORTANTE: Como en Github tratamos con un tema de maximo permitido por archivos de 100 MB, para el tema del dataset, como son archivos del orden de los GB, comprimí los datos y los voy a compartir en el correo electrónico en el que envío esta prueba. Un archivo es clean_data.csv que se puede ubicar en la raíz del proyecto y el otro archivo es dataset.csv que se debe ubicar en la ruta data/raw/clean_data.csv porque los scripts están programados para capturarlo de ahi. Adicionalmente, para adquirir los archivos del modelo y el preprocesamiento es necesario ejecutar los scripts, ya que no los pude incluir. Por esa misma razón se rompe el CI/CD de Github, porque no encuentra los archivos, y sinceramente no tuve mas tiempo para poder arreglarlo. Hubiera necesitado unas horas mas pero ya mis otras responsabilidades no me dieron espacio para dejar esa parte totalmente funcional. Pero son problemas sencillos de solucionar. Sin embargo pueden entrar a verificar el archivo ci-cd-pipeline.yml que esta en .github/workflows/ para ver toda la configuración. Gracias por la oportunidad.

### Flujo del proyecto

#### 1. Análisis exploratorio de los datos y verificación de viabilidad del entrenamiento del modelo:

Al interior de la carpeta notebooks/ encontramos un archivo llamado EDA.ipynb en donde se encuentra el acercamiento inicial con el problema, en donde se preparan los datos, se preprocesan, se hace un       análisis exploratorio y se hace una prueba de entrenamiento de un modelo inicial como validación de que es viable desarrollar scripts productivos para atacar el problema. Al interior de las casillas de codigo   se van encontrando los respectivos comentarios a lo largo del avance del experimento. Al final del notebook se encuentran algunas recomendaciones, entre esas, el uso de modelos pre-entrenados y una prueba en otro notebook llamado prueba_transformers.ipynb en donde se hace uso de HuggingFace y un modelo pre-entrenado que promete muchos mejores resultados, ademas de poder lidiar con múltiples lenguajes.

Una vez viable el desarrollo, y hechas las pruebas de concepto, se procede a desarrollar código productivo para reunir los pasos ejecutados en el EDA.ipynb pero de manera formal y reproducible.

#### ADVERTENCIA: Como el dataset solicitado es superior a 1 millon de filas, las demoras en terminos de ejecución y computación son altas para generar los archivos (horas).

Procedo a explicar el flujo de los scripts en forma de paso a paso:

2. Preprocesamiento de Datos (preprocesamiento.py)
   
     - Limpieza y normalización de texto.
     - Lematización con spaCy y eliminación de stopwords con NLTK.
     - Conversión a vectores TF-IDF.
     - Codificación de etiquetas de sentimiento.
     - Guardado de datos preprocesados y transformadores (vectorizer.pkl, encoder.pkl).

  Salida generada:

    data/processed/X.csv (Matriz de características)
    data/processed/y.csv (Etiquetas de sentimiento)
    data/processed/vectorizer.pkl (Vectorizador TF-IDF)
    data/processed/encoder.pkl (Codificador de etiquetas)

3. Entrenamiento del Modelo (training.py)

    - Carga de los datos preprocesados (X.csv, y.csv).
    - División en entrenamiento y prueba (train_test_split).
    - Selección de modelo (RandomForest, SVM o XGBoost).
    - Optimización de hiperparámetros con RandomizedSearchCV.
    - Evaluación con accuracy, precision, recall, f1-score.
    - Guardado del modelo entrenado (model.pkl).
  
Salida generada:

    data/models/model.pkl (Modelo entrenado)
    data/models/training_results.json (Métricas de evaluación)

4. Inferencia del Modelo (inferencia.py)

    - Carga del modelo (model.pkl), vectorizador (vectorizer.pkl) y encoder (encoder.pkl).
    - Conversión de texto en características con TF-IDF.
    - Predicción de sentimiento (positivo, neutro, negativo).
    - Predicción individual y en batch desde un CSV.

Salida generada (si se usa un CSV):

    data/predictions.csv (Predicciones en batch)


## Instalación y Configuración
1. Clonar el Repositorio

         git clone https://github.com/juan-arbelaez/prueba_ml.git
         cd prueba_ml

2. Crear y Activar un Entorno Virtual
   
        python -m venv venv
        source venv/bin/activate  # En Mac/Linux
        venv\Scripts\activate     # En Windows

3. Instalar Dependencias

        pip install -r requirements.txt
   
OPCIONAL: (En caso de no descargarse correctamente con el requirements.txt)
4. Descargar Modelos de NLP (spaCy)

    python -m spacy download en_core_web_sm


## Ejecución del Pipeline
NOTA: Debe estar ubicado el archivo de entrada en data/raw/dataset.csv

1. Ejecutar Preprocesamiento

       python src/preprocessing.py

2. Entrenar el Modelo

        python src/training.py
   
OPCIONAL: Puedes elegir el modelo con:

    python src/training.py --model random_forest
    python src/training.py --model svm
    python src/training.py --model xgboost
    
3️. Realizar Inferencia

    python src/inference.py
    
- Predicción desde CSV en batch:

      python src/inference.py --input data/test_samples.csv --output data/predictions.csv
  
## Pruebas Automatizadas

Se han implementado tests con pytest para garantizar la calidad del código en cada fase del pipeline.

1. Ejecutar Tests Localmente

       pytest tests/

## Integración y Despliegue con GitHub Actions

Este proyecto cuenta con CI/CD/CT (Integración, Despliegue y Testing Continuo) con GitHub Actions.

### Flujo en .github/workflows/ci-cd-pipeline.yml

1️. CI (Continuous Integration)
   
  - Analiza el código con flake8.
  - Ejecuta los tests con pytest.

2️. CD (Continuous Deployment)

  - Ejecuta preprocessing.py y training.py.
  - Guarda el modelo entrenado (model.pkl, vectorizer.pkl, encoder.pkl).

3. CT (Continuous Testing)
  
  - Descarga los artefactos generados.
  - Ejecuta inference.py y valida predicciones.

Contacto: Si tienes dudas o sugerencias, contactame a arbelaezr.juan@gmail.com
