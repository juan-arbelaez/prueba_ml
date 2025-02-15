# Módulos estándar de Python
import os
import logging
import pickle

# Módulos de terceros (instalados con pip)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict, Any, Union

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Definir rutas
MODEL_DIR = "data/models"
DATA_DIR = "data/processed"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(DATA_DIR, "vectorizer.pkl")
ENCODER_PATH = os.path.join(DATA_DIR, "encoder.pkl")

def load_model() -> Any:
    """Carga el modelo entrenado desde un archivo pickle."""
    try:
        logger.info("Cargando modelo entrenado...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info("Modelo cargado correctamente.")
        return model
    except FileNotFoundError as e:
        logger.error(f"Error: No se encontró el archivo del modelo: {e}")
        raise

def load_transformers() -> Tuple[Any, Any]:
    """Carga el vectorizador y el encoder desde archivos pickle."""
    try:
        logger.info("Cargando vectorizador y encoder...")
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)

        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)

        logger.info("Vectorizador y encoder cargados correctamente.")
        return vectorizer, encoder
    except FileNotFoundError as e:
        logger.error(f"Error: No se encontraron los archivos del vectorizador o encoder: {e}")
        raise

def preprocess_text(texts: List[str], vectorizer) -> csr_matrix:
    """
    Aplica el vectorizador a una lista de textos para convertirlos en características numéricas.

    Args:
        texts (List[str]): Lista de textos sin procesar.
        vectorizer: Objeto TfidfVectorizer previamente entrenado.

    Returns:
        csr_matrix: Matriz dispersa transformada.
    """
    try:
        logger.info("Vectorizando texto de entrada...")
        X = vectorizer.transform(texts)  # Mantiene formato disperso para optimizar memoria
        return X
    except Exception as e:
        logger.error(f"Error en la vectorización del texto: {e}")
        raise

def predict(texts: List[str]) -> List[Tuple[str, str]]:
    """
    Realiza predicciones de sentimiento sobre una lista de textos.

    Args:
        texts (List[str]): Lista de textos de entrada.

    Returns:
        List[Tuple[str, str]]: Lista de predicciones con (texto, categoría de sentimiento).
    """
    try:
        model = load_model()
        vectorizer, encoder = load_transformers()

        X = preprocess_text(texts, vectorizer)

        predictions = model.predict(X)
        labels = encoder.inverse_transform(predictions)  # Convertir de números a etiquetas

        return list(zip(texts, labels))
    except Exception as e:
        logger.error(f"Error en la predicción: {e}")
        raise

def predict_from_csv(input_path: str, output_path: Union[str, None] = None) -> pd.DataFrame:
    """
    Realiza predicciones en batch a partir de un archivo CSV y guarda los resultados.

    Args:
        input_path (str): Ruta del archivo CSV de entrada (debe contener una columna 'text').
        output_path (Union[str, None], opcional): Ruta del archivo CSV de salida. Si es None, no guarda en archivo.

    Returns:
        pd.DataFrame: DataFrame con los textos y sus predicciones.
    """
    try:
        logger.info(f"Cargando datos desde {input_path}...")
        df = pd.read_csv(input_path)

        if "text" not in df.columns:
            raise ValueError("El archivo CSV debe contener una columna llamada 'text'.")

        texts = df["text"].astype(str).tolist()  # Convertir a lista de strings
        predictions = predict(texts)

        df["sentiment"] = [label for _, label in predictions]  # Agregar predicciones

        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Predicciones guardadas en {output_path}")

        return df
    except FileNotFoundError:
        logger.error(f"Error: No se encontró el archivo {input_path}")
        raise
    except Exception as e:
        logger.error(f"Error en la predicción en batch: {e}")
        raise

def main():
    """
    Función principal que ejecuta la inferencia con ejemplos de prueba y batch.
    """
    try:
        logger.info("Ejecutando inferencia en ejemplos de prueba...")

        test_texts = [
            "The food was amazing, I loved it!",
            "The service was terrible, I am never coming back.",
            "It was an average experience, nothing special.",
            "Excelente servicio y comida deliciosa.",
            "No me gustó para nada, muy malo.",
        ]

        predictions = predict(test_texts)

        for text, label in predictions:
            logger.info(f"Texto: {text} -> Predicción: {label}")

        # Prueba con CSV
        input_csv = "data/test_samples.csv"
        output_csv = "data/predictions.csv"

        if os.path.exists(input_csv):
            predict_from_csv(input_csv, output_csv)

    except Exception as e:
        logger.error(f"Error general en la ejecución: {e}", exc_info=True)

if __name__ == "__main__":
    main()
