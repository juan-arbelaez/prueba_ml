import sys
import os

# Agregar `src/` al `PYTHONPATH`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import os
import pytest
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from src.inferencia import (
    load_model,
    load_transformers,
    preprocess_text,
    predict,
    predict_from_csv
)

@pytest.fixture
def mock_model(tmp_path):
    """ Crea un modelo RandomForest de prueba y lo guarda en un archivo pickle. """
    model = RandomForestClassifier(n_estimators=10)
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, size=(50,))
    model.fit(X_train, y_train)

    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return str(model_path)

@pytest.fixture
def mock_transformers(tmp_path):
    """ Crea archivos de vectorizador y encoder simulados. """
    vectorizer = TfidfVectorizer()
    encoder = LabelEncoder()

    vectorizer.fit(["sample text", "another example"])
    encoder.fit(["positivo", "negativo"])

    vectorizer_path = tmp_path / "vectorizer.pkl"
    encoder_path = tmp_path / "encoder.pkl"

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)

    return str(vectorizer_path), str(encoder_path)

@pytest.fixture
def mock_texts():
    """ Retorna textos de prueba. """
    return ["Great food and excellent service!", "Worst experience ever, never coming back."]

def test_load_model(mock_model):
    """Verifica que `load_model()` carga el modelo correctamente."""
    os.environ["MODEL_PATH"] = mock_model  # Sobreescribir la ruta del modelo
    model = load_model()
    assert model is not None
    assert hasattr(model, "predict")  # Debe tener el método predict

def test_load_transformers(mock_transformers):
    """Prueba que `load_transformers()` carga el vectorizador y el encoder correctamente."""
    os.environ["VECTORIZER_PATH"], os.environ["ENCODER_PATH"] = mock_transformers
    vectorizer, encoder = load_transformers()

    assert vectorizer is not None
    assert encoder is not None
    assert hasattr(vectorizer, "transform")  # Debe poder transformar texto
    assert hasattr(encoder, "inverse_transform")  # Debe poder decodificar etiquetas

def test_preprocess_text(mock_transformers, mock_texts):
    """Verifica que `preprocess_text()` vectoriza correctamente los textos de entrada."""
    os.environ["VECTORIZER_PATH"], _ = mock_transformers
    vectorizer, _ = load_transformers()
    
    X = preprocess_text(mock_texts, vectorizer)
    assert isinstance(X, csr_matrix)  # Debe ser una matriz dispersa
    assert X.shape[0] == len(mock_texts)  # Debe tener el mismo número de filas que textos

def test_predict(mock_model, mock_transformers, mock_texts):
    """Prueba que `predict()` devuelve etiquetas de sentimiento correctamente."""
    os.environ["MODEL_PATH"] = mock_model
    os.environ["VECTORIZER_PATH"], os.environ["ENCODER_PATH"] = mock_transformers

    predictions = predict(mock_texts)
    
    assert len(predictions) == len(mock_texts)  # La salida debe coincidir con la entrada
    assert all(isinstance(label, str) for _, label in predictions)  # Las etiquetas deben ser strings

@pytest.fixture
def mock_csv(tmp_path):
    """Crea un archivo CSV de prueba con textos."""
    csv_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({"text": ["Amazing experience!", "Not good at all."]})
    df.to_csv(csv_path, index=False)
    return str(csv_path)

def test_predict_from_csv(mock_model, mock_transformers, mock_csv, tmp_path):
    """Prueba que `predict_from_csv()` procesa correctamente un archivo CSV y guarda los resultados."""
    os.environ["MODEL_PATH"] = mock_model
    os.environ["VECTORIZER_PATH"], os.environ["ENCODER_PATH"] = mock_transformers

    output_csv = tmp_path / "predictions.csv"
    df = predict_from_csv(mock_csv, output_csv)

    assert "sentiment" in df.columns  # La columna de predicción debe existir
    assert df.shape[0] > 0  # Debe haber datos en el DataFrame
    assert os.path.exists(output_csv)  # El archivo de salida debe existir