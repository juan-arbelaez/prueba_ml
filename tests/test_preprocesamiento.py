import sys
import os

# Agregar `src/` al `PYTHONPATH`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import os
import pytest
import pandas as pd
import pickle
from src.preprocesamiento import (
    normalize_text,
    clean_text,
    clean_word_list,
    assign_sentiment,
    process_pipeline
)

@pytest.fixture
def sample_text():
    return "¡Hola! ¿Cómo estás? Espero que estés bien."

@pytest.fixture
def sample_dataframe():
    """Crea un DataFrame de prueba con columnas 'text' y 'stars'."""
    data = {
        "text": [
            "I love this restaurant, the food is amazing!",
            "Terrible service, I will never come back.",
            "It was an average experience, nothing special."
        ],
        "stars": [5, 1, 3]
    }
    return pd.DataFrame(data)

def test_normalize_text(sample_text):
    """Verifica que la normalización elimina tildes y caracteres especiales."""
    normalized_text = normalize_text(sample_text)
    assert "¡" not in normalized_text
    assert "¿" not in normalized_text
    assert "é" not in normalized_text  # Debe convertirse en 'e'

def test_clean_text(sample_text):
    """Verifica que `clean_text` elimine puntuación y stopwords."""
    cleaned_words = clean_text(sample_text)
    assert "hola" in cleaned_words  # Debe estar lematizado
    assert "cómo" not in cleaned_words  # Es una stop word y debe eliminarse
    assert "estás" not in cleaned_words  # También es una stop word

def test_clean_word_list():
    """Prueba que `clean_word_list` elimina espacios en blanco y números."""
    words = ["hello", "world", "123", "", "  ", "!"]
    cleaned = clean_word_list(words)
    assert "hello" in cleaned
    assert "world" in cleaned
    assert "123" not in cleaned  # Debe eliminar números
    assert "" not in cleaned  # Debe eliminar cadenas vacías
    assert "!" not in cleaned  # Debe eliminar puntuación

def test_assign_sentiment():
    """Prueba que `assign_sentiment` asigna correctamente el sentimiento."""
    assert assign_sentiment(5) == "positivo"
    assert assign_sentiment(4) == "positivo"
    assert assign_sentiment(3) == "neutro"
    assert assign_sentiment(2) == "negativo"
    assert assign_sentiment(1) == "negativo"

def test_process_pipeline(sample_dataframe, tmp_path):
    """
    Prueba que `process_pipeline` ejecuta correctamente el preprocesamiento
    y genera los archivos necesarios.
    """
    # Crear directorio temporal para evitar modificaciones en la estructura real
    data_dir = tmp_path / "data/processed"
    os.makedirs(data_dir, exist_ok=True)

    # Ejecutar pipeline en un DataFrame de prueba
    X, y = process_pipeline(sample_dataframe)

    # Validar que las salidas sean correctas
    assert X.shape[0] == sample_dataframe.shape[0]  # Mismo número de filas
    assert y.shape[0] == sample_dataframe.shape[0]  # Mismo número de etiquetas

    # Verificar que los archivos han sido guardados correctamente
    assert os.path.exists(data_dir / "X.csv")
    assert os.path.exists(data_dir / "y.csv")
    assert os.path.exists(data_dir / "vectorizer.pkl")
    assert os.path.exists(data_dir / "encoder.pkl")

    # Cargar y verificar que vectorizador y encoder son objetos válidos
    with open(data_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
        assert hasattr(vectorizer, "transform")  # Debe tener el método transform

    with open(data_dir / "encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
        assert hasattr(encoder, "transform")  # Debe tener el método transform
