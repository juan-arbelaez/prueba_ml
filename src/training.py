# Módulos estándar de Python
import os
import logging
import pickle
import json

# Módulos de terceros (instalados con pip)
import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

# Tipado estático
from typing import Tuple, Dict, Any

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Definir ruta base
DATA_DIR = "data/processed"
MODEL_DIR = "data/models"
RESULTS_FILE = os.path.join(MODEL_DIR, "training_results.json")

def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga los datos preprocesados desde archivos CSV.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (características) y y (etiquetas) como arreglos de NumPy.

    Raises:
        FileNotFoundError: Si los archivos de datos no existen.
    """
    try:
        logger.info("Cargando datos preprocesados...")
        X = pd.read_csv(os.path.join(DATA_DIR, "X.csv")).values
        y = pd.read_csv(os.path.join(DATA_DIR, "y.csv")).values.ravel()
        return X, y
    except FileNotFoundError as e:
        logger.error(f"Error: No se encontraron los archivos de datos preprocesados: {e}")
        raise

def load_transformers() -> Tuple:
    """
    Carga el vectorizador y el encoder desde archivos.

    Returns:
        Tuple: Vectorizador y codificador de etiquetas.

    Raises:
        FileNotFoundError: Si los archivos no existen.
    """
    try:
        logger.info("Cargando vectorizador y encoder...")
        with open(os.path.join(DATA_DIR, "vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)

        with open(os.path.join(DATA_DIR, "encoder.pkl"), "rb") as f:
            encoder = pickle.load(f)

        return vectorizer, encoder
    except FileNotFoundError as e:
        logger.error(f"Error: No se encontraron los archivos del vectorizador o encoder: {e}")
        raise

def get_model(model_name: str):
    """
    Devuelve el modelo seleccionado.

    Args:
        model_name (str): Nombre del modelo a entrenar.

    Returns:
        Un modelo de scikit-learn o XGBoost.
    """
    models = {
        "random_forest": RandomForestClassifier(random_state=42),
        "svm": SVC(probability=True, random_state=42),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }

    if model_name not in models:
        logger.error(f"Modelo '{model_name}' no soportado. Opciones: {list(models.keys())}")
        raise ValueError(f"Modelo '{model_name}' no es válido. Opciones: {list(models.keys())}")

    return models[model_name]

def tune_hyperparameters(model_name: str, X_train: np.ndarray, y_train: np.ndarray):
    """
    Aplica RandomizedSearchCV para encontrar los mejores hiperparámetros.

    Args:
        model_name (str): Nombre del modelo a optimizar.
        X_train (np.ndarray): Características de entrenamiento.
        y_train (np.ndarray): Etiquetas de entrenamiento.

    Returns:
        Modelo optimizado.
    """
    logger.info(f"Iniciando búsqueda de hiperparámetros para {model_name}...")

    param_distributions = {
        "random_forest": {
            "n_estimators": randint(50, 200),
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": randint(2, 10),
            "min_samples_leaf": randint(1, 5),
            "bootstrap": [True, False]
        },
        "svm": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"]
        },
        "xgboost": {
            "n_estimators": randint(50, 200),
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": randint(3, 10),
            "subsample": [0.6, 0.8, 1.0]
        }
    }

    base_model = get_model(model_name)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions[model_name],
        n_iter=20,
        cv=3,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    logger.info(f"Mejores hiperparámetros encontrados: {search.best_params_}")
    return search.best_estimator_, search.best_params_

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Evalúa el modelo entrenado y devuelve métricas.

    Args:
        model: Modelo entrenado.
        X_test (np.ndarray): Características de prueba.
        y_test (np.ndarray): Etiquetas de prueba.

    Returns:
        Dict[str, Any]: Diccionario con métricas de evaluación.
    """
    logger.info("Evaluando modelo en conjunto de prueba...")
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    logger.info(f"Exactitud del modelo: {metrics['accuracy']:.4f}")
    logger.info(f"Precisión: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-score: {metrics['f1_score']:.4f}")

    return metrics

def save_results(results: Dict[str, Any]):
    """
    Guarda los resultados del entrenamiento en un archivo JSON.

    Args:
        results (Dict[str, Any]): Diccionario con los resultados y métricas.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Resultados guardados en {RESULTS_FILE}")

def main(model_name: str):
    """
    Función principal que ejecuta el pipeline de entrenamiento.

    Args:
        model_name (str): Nombre del modelo a entrenar.
    """
    try:
        X, y = load_data()
        vectorizer, encoder = load_transformers()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model, best_params = tune_hyperparameters(model_name, X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        results = {"model": model_name, "best_params": best_params, "metrics": metrics}
        save_results(results)

        with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
            pickle.dump(model, f)

        logger.info("Entrenamiento finalizado con éxito.")

    except Exception as e:
        logger.error(f"Error general en la ejecución: {e}", exc_info=True)

if __name__ == "__main__":
    main("random_forest")  # Cambiar a "svm" o "xgboost" para entrenar otros modelos
