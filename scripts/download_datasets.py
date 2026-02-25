"""
Script para descargar datasets de Kaggle usando kagglehub.
No requiere configuración de credenciales manual.
"""

import os
import pandas as pd
import kagglehub

def descargar_datasets():
    """
    Descarga los 3 datasets de Kaggle usando kagglehub.
    
    Datasets:
    1. Resume Screening - CVs clasificados por tipo de profesional
    2. Campus Recruitment - Perfiles de estudiantes y colocación
    3. Student Essays - Ensayos con puntuaciones de calidad
    """
    
    directorio_datos = "./data"
    os.makedirs(directorio_datos, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"📥 DESCARGANDO DATASETS DE KAGGLE")
    print(f"{'='*80}\n")
    
    # Dataset 1: Resume Screening
    print(f"1️⃣ Descargando Resume Screening Dataset...")
    try:
        path = kagglehub.dataset_download("mfaisalqureshi/resume-screening-dataset")
        # Buscar archivo CSV principal
        for archivo in os.listdir(path):
            if archivo.endswith('.csv'):
                df = pd.read_csv(os.path.join(path, archivo))
                df_resumenes = df.head(2000)  # Limitar a 2000 registros
                df_resumenes.to_csv(f"{directorio_datos}/resume_screening.csv", index=False)
                print(f"   ✅ Descargado: resume_screening.csv")
                print(f"   Registros: {len(df_resumenes)}")
                break
    except Exception as e:
        print(f"   ❌ Error descargando Resume Screening: {e}")
    
    # Dataset 2: Campus Recruitment
    print(f"\n2️⃣ Descargando Campus Recruitment Dataset...")
    try:
        path = kagglehub.dataset_download("benroshan/campus-recruitment-data")
        # Buscar archivo CSV principal
        for archivo in os.listdir(path):
            if archivo.endswith('.csv'):
                df = pd.read_csv(os.path.join(path, archivo))
                df.to_csv(f"{directorio_datos}/campus_recruitment.csv", index=False)
                print(f"   ✅ Descargado: campus_recruitment.csv")
                print(f"   Registros: {len(df)}")
                break
    except Exception as e:
        print(f"   ❌ Error descargando Campus Recruitment: {e}")
    
    # Dataset 3: Student Performance / Essays
    print(f"\n3️⃣ Intentando descargar Student Essays Dataset...")
    try:
        # Alternativa: Student Performance in Exams (datos públicos)
        path = kagglehub.dataset_download("spscientist/students-performance-in-exams")
        for archivo in os.listdir(path):
            if archivo.endswith('.csv'):
                df = pd.read_csv(os.path.join(path, archivo))
                df.to_csv(f"{directorio_datos}/student_performance.csv", index=False)
                print(f"   ✅ Descargado: student_performance.csv")
                print(f"   Registros: {len(df)}")
                break
    except Exception as e:
        print(f"   ❌ Error descargando Student Performance: {e}")
    
    print(f"\n{'='*80}")
    print(f"✅ DESCARGA COMPLETADA")
    print(f"{'='*80}\n")
    print(f"Datasets guardados en: {os.path.abspath(directorio_datos)}/")
    print(f"Archivos:")
    for archivo in os.listdir(directorio_datos):
        print(f"  - {archivo}")

if __name__ == "__main__":
    print("\n⚠️  IMPORTANTE: Esta es la primera ejecución")
    print("Se descargarán los datasets de Kaggle (~500MB total)")
    print("Puede tomar 2-5 minutos según tu conexión\n")
    
    respuesta = input("¿Continuar con la descarga? (s/n): ")
    if respuesta.lower() == 's':
        descargar_datasets()
    else:
        print("Descargas cancelada.")
