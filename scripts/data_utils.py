"""
Utilidades para cargar datasets reales de Hugging Face
Datasets utilizados:
- IMDB: https://huggingface.co/datasets/imdb
- AG News: https://huggingface.co/datasets/ag_news
- DBpedia: https://huggingface.co/datasets/dbpedia_14 (https://www.dbpedia.org/)
"""

import pandas as pd
from datasets import Dataset, DatasetDict


def cargar_dataset_csv_personalizado(ruta_csv, columna_texto="texto", columna_etiqueta="etiqueta", tamaño_entrenamiento=400, tamaño_prueba=100):
    """
    Carga dataset personalizado desde CSV (útil para agregar datasets propios).
    
    Args:
        ruta_csv: Ruta al archivo CSV
        columna_texto: Nombre de la columna de texto
        columna_etiqueta: Nombre de la columna de etiquetas
        tamaño_entrenamiento: Cantidad de datos para training
        tamaño_prueba: Cantidad de datos para testing
    
    Returns:
        DatasetDict con splits entrenamiento/prueba
    """
    
    datos = pd.read_csv(ruta_csv)
    
    # Crear Dataset
    conjunto_datos = Dataset.from_pandas(datos[[columna_texto, columna_etiqueta]])
    
    # Split entrenamiento/prueba
    division = conjunto_datos.train_test_split(test_size=tamaño_prueba, seed=42)
    
    # Limitar tamaños
    entrenamiento = division["train"].select(range(min(tamaño_entrenamiento, len(division["train"]))))
    prueba = division["test"].select(range(min(tamaño_prueba, len(division["test"]))))
    
    diccionario_conjunto = DatasetDict({"train": entrenamiento, "test": prueba})
    
    print(f"✅ Dataset cargado desde {ruta_csv}")
    print(f"   Muestras entrenamiento: {len(entrenamiento)}")
    print(f"   Muestras prueba: {len(prueba)}")
    
    return diccionario_conjunto


def validar_estructura_dataset(diccionario_conjunto):
    """Valida que un dataset tenga estructura correcta."""
    
    if not isinstance(diccionario_conjunto, DatasetDict):
        raise TypeError("El dataset debe ser un DatasetDict")
    
    if "train" not in diccionario_conjunto or "test" not in diccionario_conjunto:
        raise ValueError("El dataset debe tener splits 'train' y 'test'")
    
    columnas_requeridas = {"texto", "etiqueta"}
    columnas_entrenamiento = set(diccionario_conjunto["train"].column_names)
    
    if not columnas_requeridas.issubset(columnas_entrenamiento):
        raise ValueError(f"El dataset debe tener columnas: {columnas_requeridas}")
    
    print("✅ Estructura del dataset validada")


if __name__ == "__main__":
    # Ejemplo: cargar un dataset personalizado desde CSV
    # conjunto = cargar_dataset_csv_personalizado("./data/mi_dataset.csv")
    # validar_estructura_dataset(conjunto)
    print("📚 Utilidades de datasets disponibles")
    print("Datasets reales cargados desde HuggingFace:")
    print("- IMDB (2 clases)")
    print("- AG News (4 clases)")
    print("- DBpedia (14 clases)")

