"""
Script para descargar datasets de Kaggle usando kagglehub.
No requiere configuración de credenciales manual.
"""

import os
import argparse
import pandas as pd
import kagglehub


def _archivo_existe(ruta_archivo):
    """Retorna True si el archivo ya existe y no está vacío."""
    return os.path.exists(ruta_archivo) and os.path.getsize(ruta_archivo) > 0


DATASETS = [
    {
        "nombre": "resume_screening",
        "archivo": "resume_screening.csv",
        "kaggle_id": "mfaisalqureshi/resume-screening-dataset",
        "url": "https://www.kaggle.com/datasets/mfaisalqureshi/resume-screening-dataset",
    },
    {
        "nombre": "campus_recruitment",
        "archivo": "campus_recruitment.csv",
        "kaggle_id": "benroshan/campus-recruitment-data",
        "url": "https://www.kaggle.com/datasets/benroshan/campus-recruitment-data",
    },
    {
        "nombre": "student_performance",
        "archivo": "student_performance.csv",
        "kaggle_id": "spscientist/students-performance-in-exams",
        "url": "https://www.kaggle.com/datasets/spscientist/students-performance-in-exams",
    },
]


def estado_datasets_locales(directorio_datos="./data"):
    """
    Verifica cuáles CSV de talento estudiantil ya existen en disco.

    Returns:
        tuple(list, list): (disponibles, faltantes)
    """
    os.makedirs(directorio_datos, exist_ok=True)
    disponibles = []
    faltantes = []

    for dataset in DATASETS:
        ruta = os.path.join(directorio_datos, dataset["archivo"])
        if _archivo_existe(ruta):
            disponibles.append(dataset)
        else:
            faltantes.append(dataset)

    return disponibles, faltantes


def _descargar_dataset_kaggle(dataset, directorio_datos):
    """Descarga un dataset de Kaggle y guarda el primer CSV encontrado."""
    ruta_salida = os.path.join(directorio_datos, dataset["archivo"])
    path = kagglehub.dataset_download(dataset["kaggle_id"])

    for archivo in os.listdir(path):
        if archivo.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, archivo))
            if dataset["nombre"] == "resume_screening":
                # Limitar tamaño para iteración más rápida en local.
                df = df.head(2000)
            df.to_csv(ruta_salida, index=False)
            return len(df)

    raise FileNotFoundError(f"No se encontró CSV en el dataset {dataset['kaggle_id']}")


def descargar_datasets(intentar_kaggle=False):
    """
    Prepara carpeta local de datos y valida presencia de CSV.
    Opcionalmente descarga faltantes desde Kaggle.
    
    Datasets:
    1. Resume Screening - CVs clasificados por tipo de profesional
    2. Campus Recruitment - Perfiles de estudiantes y colocación
    3. Student Performance - Rendimiento académico
    """
    
    directorio_datos = "./data"
    os.makedirs(directorio_datos, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"📁 PREPARANDO DATASETS DE TALENTO ESTUDIANTIL")
    print(f"{'='*80}\n")

    disponibles, faltantes = estado_datasets_locales(directorio_datos)

    if disponibles:
        print("✅ CSV ya disponibles:")
        for dataset in disponibles:
            print(f"  - {dataset['archivo']}")
        print("")

    if not faltantes:
        print("✅ Ya están los 3 CSV requeridos. No hay nada que descargar.\n")
    else:
        print("⚠️ CSV faltantes:")
        for dataset in faltantes:
            print(f"  - {dataset['archivo']}")
            print(f"    URL: {dataset['url']}")
        print("")

        if not intentar_kaggle:
            print("ℹ️ Modo manual: descarga esos CSV y guárdalos en ./data con esos nombres.")
            print("ℹ️ Si quieres intentar descarga automática por Kaggle, ejecuta con --kaggle.\n")
        else:
            print("🔄 Intentando descarga automática por Kaggle para los faltantes...\n")
            for idx, dataset in enumerate(faltantes, start=1):
                print(f"{idx}️⃣ Descargando {dataset['nombre']}...")
                try:
                    total = _descargar_dataset_kaggle(dataset, directorio_datos)
                    print(f"   ✅ Descargado: {dataset['archivo']}")
                    print(f"   Registros: {total}")
                except Exception as e:
                    print(f"   ❌ Error descargando {dataset['nombre']}: {e}")
    
    print(f"\n{'='*80}")
    print(f"✅ PROCESO COMPLETADO")
    print(f"{'='*80}\n")
    print(f"Datasets guardados en: {os.path.abspath(directorio_datos)}/")
    print(f"Archivos:")
    for archivo in os.listdir(directorio_datos):
        print(f"  - {archivo}")

if __name__ == "__main__":
    analizador = argparse.ArgumentParser(
        description="Prepara y valida CSVs de datasets de talento estudiantil."
    )
    analizador.add_argument(
        "--kaggle",
        action="store_true",
        help="Intenta descargar automáticamente los CSV faltantes desde Kaggle.",
    )
    argumentos = analizador.parse_args()
    descargar_datasets(intentar_kaggle=argumentos.kaggle)
