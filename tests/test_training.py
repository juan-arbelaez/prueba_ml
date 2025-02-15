import sys
import os

# Agregar `src/` al `PYTHONPATH`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import os
import json
import pytest
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.training import (
    load_data,
    load_transformers,
    get_model,
    tune_hyperparameters,
    evaluate_model,
    save_results
)

@pytest.fixture
def mock_data(tmp_path):
    """ Crea archivos de datos preprocesados de prueba. """
    data_dir = tmp_path / "data/processed"
    os.makedirs(data_dir, exist_ok=True)

    # Crear datos simulados
    X = np.random.rand(100, 10)  # 100 muestras, 10 características
    y = np.random.randint(0, 2, size=(100,))  # 100 etiquetas binarias

    # Guardar como CSV
    pd.DataFrame(X).to_csv(data_dir / "X.csv", index=False)
    pd.DataFrame(y, columns=["sentiment"]).to_csv(data_dir / "y.csv", index=False)

    return str(data_dir)

@pytest.fixture
def mock_transformers(tmp_path):
    """ Crea archivos pickle para el vectorizador y encoder. """
    data_dir = tmp_path / "data/processed"
    os.makedirs(data_dir, exist_ok=True)

    vectorizer = RandomForestClassifier()  # Simulación de objeto con método `transform`
    encoder = {"positivo": 1, "negativo": 0}  # Simulación de un encoder

    with open(data_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(data_dir / "encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    return str(data_dir)

def test_load_data(mock_data):
    """Verifica que load_data() carga correctamente los archivos CSV."""
    X, y = load_data(mock_data)

    assert X.shape == (100, 10)  # Debe tener 100 filas y 10 columnas
    assert y.shape == (100,)  # Debe tener 100 etiquetas

def test_load_transformers(mock_transformers):
    """Prueba que load_transformers() carga correctamente los objetos pickle."""
    vectorizer, encoder = load_transformers(mock_transformers)

    assert vectorizer is not None
    assert encoder is not None

def test_get_model():
    """Prueba que get_model() devuelve los modelos correctos."""
    rf_model = get_model("random_forest")
    assert isinstance(rf_model, RandomForestClassifier)

    # Modelo no existente debería lanzar error
    with pytest.raises(ValueError):
        get_model("modelo_inexistente")

def test_tune_hyperparameters():
    """Prueba que tune_hyperparameters() devuelve un modelo optimizado."""
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, size=(50,))

    model, best_params = tune_hyperparameters("random_forest", X_train, y_train)

    assert isinstance(model, RandomForestClassifier)
    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params  # Verificar que tiene hiperparámetros optimizados

def test_evaluate_model():
    """Prueba que evaluate_model() devuelve métricas correctamente."""
    model = RandomForestClassifier(n_estimators=10)
    X_test = np.random.rand(20, 10)
    y_test = np.random.randint(0, 2, size=(20,))

    model.fit(X_test, y_test)
    metrics = evaluate_model(model, X_test, y_test)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert isinstance(metrics["accuracy"], float)

def test_save_results(tmp_path):
    """Prueba que save_results() guarda un archivo JSON correctamente."""
    results = {
        "model": "random_forest",
        "best_params": {"n_estimators": 100},
        "metrics": {"accuracy": 0.85}
    }
    file_path = tmp_path / "training_results.json"
    
    save_results(results, str(file_path))
    
    assert os.path.exists(file_path)

    with open(file_path, "r") as f:
        saved_results = json.load(f)
    
    assert saved_results["model"] == "random_forest"
    assert saved_results["best_params"]["n_estimators"] == 100
    assert saved_results["metrics"]["accuracy"] == 0.85
