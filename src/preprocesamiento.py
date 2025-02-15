# Módulos estándar de Python
import logging
import os
import pickle
import string
import unicodedata

# Módulos de terceros (paquetes instalados con pip)
import nltk
import spacy
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Tipado estático
from typing import List, Union

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Verificar y descargar stopwords si no están disponibles
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    logger.info("Descargando stopwords de NLTK...")
    nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

# Cargar modelo de NLP de spaCy
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Modelo de spaCy cargado correctamente.")
except Exception as e:
    logger.error(f"Error al cargar el modelo de spaCy: {e}")
    raise


def normalize_text(text: str) -> str:
    """Normaliza el texto eliminando tildes y caracteres especiales."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")


def clean_text(text: str, stop_words: set[str] = STOP_WORDS) -> List[str]:
    """Limpia y normaliza un texto eliminando puntuación, acentos y stop words."""
    if not isinstance(text, str):
        logger.error("El texto de entrada no es una cadena de caracteres.")
        raise ValueError("El texto de entrada debe ser una cadena de caracteres.")

    text = normalize_text(text.lower())
    text = text.translate(str.maketrans("", "", string.punctuation))

    doc = nlp(text)
    words = [
        token.lemma_
        for token in doc
        if token.text not in stop_words and not token.is_punct
    ]
    return words


def clean_word_list(words: List[str]) -> List[str]:
    """Limpia una lista de palabras eliminando espacios en blanco, signos de puntuación y números."""
    return list(
        filter(
            None,
            map(
                lambda word: (
                    word.strip()
                    if word and word not in string.punctuation and not word.isdigit()
                    else None
                ),
                words,
            ),
        )
    )


def assign_sentiment(stars: Union[int, float]) -> str:
    """Asigna una etiqueta de sentimiento ('negativo', 'neutro' o 'positivo') basada en las estrellas."""
    if stars <= 2:
        return "negativo"
    elif stars == 3:
        return "neutro"
    else:
        return "positivo"


def process_pipeline(df: pd.DataFrame) -> tuple:
    """
    Ejecuta el preprocesamiento de datos:
    - Filtra columnas necesarias.
    - Limpia y normaliza texto.
    - Asigna sentimiento a las calificaciones.
    - Convierte texto a vectores TF-IDF.
    - Codifica etiquetas de sentimiento.
    - Guarda los datos preprocesados.

    Args:
        df (pd.DataFrame): DataFrame con columnas 'text' y 'stars'.

    Returns:
        tuple: Matriz de características (X) y etiquetas codificadas (y).
    """
    try:
        logger.info("Iniciando preprocesamiento de datos...")

        # Verificar existencia de columnas
        if "text" not in df or "stars" not in df:
            logger.error(
                "El DataFrame no contiene las columnas necesarias ('text' y 'stars')."
            )
            raise KeyError("El DataFrame debe contener las columnas 'text' y 'stars'.")

        # Filtrar solo las columnas necesarias
        df = df[["text", "stars"]].copy()

        # Aplicar limpieza al texto
        logger.info("Limpiando texto...")
        df["clean_words"] = df["text"].apply(clean_text)
        df["clean_words"] = df["clean_words"].apply(clean_word_list)

        # Asignar sentimiento
        logger.info("Asignando sentimiento...")
        df["sentimiento"] = df["stars"].apply(assign_sentiment)

        # Convertir listas de palabras a strings
        df["clean_text"] = df["clean_words"].apply(lambda words: " ".join(words))

        # Vectorización con TF-IDF
        logger.info("Vectorizando texto con TfidfVectorizer...")
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df["clean_text"])

        # Convertir etiquetas de sentimiento en números
        logger.info("Codificando etiquetas...")
        encoder = LabelEncoder()
        y = encoder.fit_transform(df["sentimiento"])

        # Crear directorio si no existe
        os.makedirs("data/processed", exist_ok=True)

        # Guardar los datos preprocesados
        pd.DataFrame(X.toarray()).to_csv("data/processed/X.csv", index=False)
        pd.DataFrame(y).to_csv("data/processed/y.csv", index=False)

        # Guardar vectorizador y encoder
        with open("data/processed/vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

        with open("data/processed/encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)

        logger.info("Pipeline ejecutado con éxito.")
        return X, y

    except Exception as e:
        logger.error(f"Error en el pipeline de procesamiento: {e}", exc_info=True)
        raise


def main():
    """Función principal que ejecuta todo el pipeline."""
    try:
        logger.info("Cargando dataset...")
        df = pd.read_csv("../data/raw/dataset.csv")

        logger.info("Ejecutando pipeline de preprocesamiento...")
        process_pipeline(df)

        logger.info("Preprocesamiento finalizado correctamente.")
    except FileNotFoundError:
        logger.error("El archivo 'data/raw/dataset.csv' no fue encontrado.")
    except Exception as e:
        logger.error(f"Error general en la ejecución: {e}", exc_info=True)


if __name__ == "__main__":
    main()
