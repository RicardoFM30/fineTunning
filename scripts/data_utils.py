"""
Utilidades para cargar datasets de talento estudiantil/juvenil desde CSV.
"""

import os
import pandas as pd
from datasets import Dataset, DatasetDict


def _normalizar_resume_screening(df):
    if "resume_text" in df.columns and "Category" in df.columns:
        df = df.rename(columns={"resume_text": "text", "Category": "label"})
        df = df[["text", "label"]].dropna()
    elif "Job Role" in df.columns:
        def crear_texto_cv(row):
            partes = []
            if "Skills" in row.index and pd.notna(row["Skills"]):
                partes.append(f"Skills: {row['Skills']}")
            if "Experience (Years)" in row.index and pd.notna(row["Experience (Years)"]):
                partes.append(f"ExperienceYears: {row['Experience (Years)']}")
            if "Education" in row.index and pd.notna(row["Education"]):
                partes.append(f"Education: {row['Education']}")
            if "Certifications" in row.index and pd.notna(row["Certifications"]):
                partes.append(f"Certifications: {row['Certifications']}")
            if "Projects Count" in row.index and pd.notna(row["Projects Count"]):
                partes.append(f"ProjectsCount: {row['Projects Count']}")
            if "AI Score (0-100)" in row.index and pd.notna(row["AI Score (0-100)"]):
                partes.append(f"AIScore: {row['AI Score (0-100)']}")
            return " ".join(partes) if partes else "Student resume"

        df["text"] = df.apply(crear_texto_cv, axis=1)
        df["label"] = df["Job Role"]
        df = df[["text", "label"]].dropna()
    else:
        raise ValueError("resume_screening.csv no tiene columnas compatibles")

    labels = df["label"].unique()
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    df["label"] = df["label"].map(label_to_id)
    return df, label_to_id


def _normalizar_campus_recruitment(df):
    if "specialisation" in df.columns and "specialization" not in df.columns:
        df = df.rename(columns={"specialisation": "specialization"})
    if "degree_t" in df.columns and "Degree" not in df.columns:
        df = df.rename(columns={"degree_t": "Degree"})

    def crear_texto_perfil(row):
        textos = []
        if "Degree" in row and pd.notna(row["Degree"]):
            textos.append(f"Carrera: {row['Degree']}")
        if "specialization" in row.index and pd.notna(row.get("specialization")):
            textos.append(f"Especialización: {row['specialization']}")
        if "cgpa" in row.index and pd.notna(row.get("cgpa")):
            textos.append(f"CGPA: {row['cgpa']}")
        if "mba_p" in row.index and pd.notna(row.get("mba_p")):
            textos.append(f"MBA%: {row['mba_p']}")
        if "etest_p" in row.index and pd.notna(row.get("etest_p")):
            textos.append(f"Etest%: {row['etest_p']}")
        if "internships" in row.index and pd.notna(row.get("internships")):
            textos.append(f"Pasantías: {row['internships']}")
        if "workex" in row.index and pd.notna(row.get("workex")):
            textos.append(f"WorkEx: {row['workex']}")
        return " ".join(textos) if textos else "Estudiante"

    df["text"] = df.apply(crear_texto_perfil, axis=1)

    if "status" in df.columns:
        df = df.rename(columns={"status": "label"})

    label_values = [label for label in df["label"].unique() if pd.notna(label)]
    label_to_id = {label: idx for idx, label in enumerate(label_values)}
    df["label"] = df["label"].map(label_to_id)
    df = df[["text", "label"]].dropna()

    return df, label_to_id


def _normalizar_student_performance(df):
    if (
        "math score" in df.columns
        and "reading score" in df.columns
        and "writing score" in df.columns
    ):
        def crear_texto_academico(row):
            textos = []
            columnas = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]
            for col in columnas:
                if col in row.index and pd.notna(row[col]):
                    textos.append(f"{col}: {row[col]}")
            return " ".join(textos) if textos else "Estudiante"

        df["text"] = df.apply(crear_texto_academico, axis=1)
        df["puntuacion_promedio"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3
    elif "G1" in df.columns and "G2" in df.columns and "G3" in df.columns:
        def crear_texto_academico(row):
            columnas_texto = [
                "school", "sex", "age", "address", "famsize", "Pstatus",
                "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
                "traveltime", "studytime", "failures", "schoolsup", "famsup",
                "paid", "activities", "internet", "romantic", "absences"
            ]
            textos = []
            for col in columnas_texto:
                if col in row.index and pd.notna(row[col]):
                    textos.append(f"{col}:{row[col]}")
            return " ".join(textos) if textos else "Estudiante"

        df["text"] = df.apply(crear_texto_academico, axis=1)
        df["puntuacion_promedio"] = (df["G1"] + df["G2"] + df["G3"]) / 3
    else:
        raise ValueError("student_performance.csv no tiene columnas compatibles")

    df["label"] = pd.cut(
        df["puntuacion_promedio"],
        bins=3,
        labels=["Bajo", "Medio", "Alto"],
        include_lowest=True,
    )

    label_to_id = {"Bajo": 0, "Medio": 1, "Alto": 2}
    df["label"] = df["label"].astype(str).map(label_to_id)
    df = df[["text", "label"]].dropna()

    return df, label_to_id


def cargar_dataset_talento_desde_csv(nombre_dataset, directorio_datos="./data", tamaño_entrenamiento=None, tamaño_prueba=None):
    """
    Carga dataset de talento estudiantil y devuelve DatasetDict + mapeo de etiquetas.

    Args:
        nombre_dataset: resume_screening | campus_recruitment | student_performance
        directorio_datos: carpeta con CSVs
        tamaño_entrenamiento: límite opcional para split train
        tamaño_prueba: límite opcional para split test

    Returns:
        tuple(DatasetDict, dict): dataset con splits train/test y label_to_id.
    """
    rutas = {
        "resume_screening": "resume_screening.csv",
        "campus_recruitment": "campus_recruitment.csv",
        "student_performance": "student_performance.csv",
    }

    if nombre_dataset not in rutas:
        raise ValueError("Dataset no soportado")

    ruta = os.path.join(directorio_datos, rutas[nombre_dataset])
    df = pd.read_csv(ruta)

    if nombre_dataset == "resume_screening":
        df, label_to_id = _normalizar_resume_screening(df)
    elif nombre_dataset == "campus_recruitment":
        df, label_to_id = _normalizar_campus_recruitment(df)
    else:
        df, label_to_id = _normalizar_student_performance(df)

    inicio_prueba = len(df) // 2
    limite_prueba = max(1, len(df) // 4)

    train_df = df.head(inicio_prueba)
    test_df = df.iloc[inicio_prueba : inicio_prueba + limite_prueba]

    if tamaño_entrenamiento is not None:
        train_df = train_df.head(tamaño_entrenamiento)
    if tamaño_prueba is not None:
        test_df = test_df.head(tamaño_prueba)

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
            "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
        }
    )

    return dataset_dict, label_to_id


def validar_estructura_dataset(diccionario_conjunto):
    """Valida estructura mínima: train/test con columnas text/label."""
    if not isinstance(diccionario_conjunto, DatasetDict):
        raise TypeError("El dataset debe ser un DatasetDict")

    if "train" not in diccionario_conjunto or "test" not in diccionario_conjunto:
        raise ValueError("El dataset debe tener splits 'train' y 'test'")

    columnas_requeridas = {"text", "label"}
    columnas_train = set(diccionario_conjunto["train"].column_names)

    if not columnas_requeridas.issubset(columnas_train):
        raise ValueError(f"El dataset debe tener columnas: {columnas_requeridas}")


if __name__ == "__main__":
    print("📚 Utilidades de datasets de talento estudiantil disponibles")
