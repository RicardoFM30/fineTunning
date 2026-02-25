"""
Script de evaluación exhaustiva: métricas, gráficas y reporte para datasets
sobre talento estudiantil/juvenil.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def validar_csv_requerido(ruta_archivo):
    """Valida que el CSV exista y tenga contenido antes de evaluar."""
    if not os.path.exists(ruta_archivo) or os.path.getsize(ruta_archivo) == 0:
        raise FileNotFoundError(
            f"❌ No se encontró el archivo requerido: {ruta_archivo}\n"
            "   Descárgalo manualmente y colócalo en ./data con ese nombre."
        )


class EvaluadorAgotador:
    """Evaluación exhaustiva de modelos entrenados."""

    def __init__(self, directorio_modelo, directorio_salida="./results", directorio_datos="./data"):
        self.directorio_modelo = directorio_modelo
        self.directorio_salida = directorio_salida
        self.directorio_datos = directorio_datos
        os.makedirs(directorio_salida, exist_ok=True)

        print(f"📂 Evaluando modelo: {directorio_modelo}")

    def cargar_modelo_y_config(self):
        """Carga modelo/tokenizador y metadata del entrenamiento."""
        self.tokenizador = AutoTokenizer.from_pretrained(self.directorio_modelo)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(self.directorio_modelo)

        archivo_config = os.path.join(self.directorio_modelo, "training_config.json")
        if os.path.exists(archivo_config):
            with open(archivo_config, "r", encoding="utf-8") as archivo:
                self.configuracion = json.load(archivo)
        else:
            self.configuracion = {}

        self.label_to_id = self.configuracion.get("label_to_id") or {}
        print("✅ Modelo y configuración cargados")

    def _cargar_dataset_local(self, nombre_dataset):
        """Carga y prepara el dataset local con el mismo esquema usado en train.py."""
        if nombre_dataset == "resume_screening":
            ruta = os.path.join(self.directorio_datos, "resume_screening.csv")
            validar_csv_requerido(ruta)
            df = pd.read_csv(ruta)

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
                raise ValueError(
                    "resume_screening.csv no tiene columnas compatibles para evaluación."
                )

            if self.label_to_id:
                df["label"] = df["label"].map(self.label_to_id)
            else:
                label_to_id = {label: idx for idx, label in enumerate(df["label"].unique())}
                df["label"] = df["label"].map(label_to_id)

            df = df.dropna(subset=["label"])

        elif nombre_dataset == "campus_recruitment":
            ruta = os.path.join(self.directorio_datos, "campus_recruitment.csv")
            validar_csv_requerido(ruta)
            df = pd.read_csv(ruta)

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

            if self.label_to_id:
                df["label"] = df["label"].map(self.label_to_id)
            else:
                label_values = [label for label in df["label"].unique() if pd.notna(label)]
                label_to_id = {label: idx for idx, label in enumerate(label_values)}
                df["label"] = df["label"].map(label_to_id)

            df = df[["text", "label"]].dropna()

        elif nombre_dataset == "student_performance":
            ruta = os.path.join(self.directorio_datos, "student_performance.csv")
            validar_csv_requerido(ruta)
            df = pd.read_csv(ruta)

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
                df["puntuacion_promedio"] = (
                    df["math score"] + df["reading score"] + df["writing score"]
                ) / 3
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
                raise ValueError(
                    "student_performance.csv no tiene columnas compatibles para evaluación."
                )

            df["label"] = pd.cut(
                df["puntuacion_promedio"],
                bins=3,
                labels=["Bajo", "Medio", "Alto"],
                include_lowest=True,
            )

            if self.label_to_id:
                df["label"] = df["label"].astype(str).map(self.label_to_id)
            else:
                df["label"] = df["label"].astype(str).map({"Bajo": 0, "Medio": 1, "Alto": 2})

            df = df[["text", "label"]].dropna()

        else:
            raise ValueError(
                "Dataset no soportado. Usa: resume_screening, campus_recruitment, student_performance"
            )

        # Mismo esquema de train.py: evaluar sobre una porción del segundo bloque del dataset.
        inicio_prueba = len(df) // 2
        tamaño_prueba = max(1, len(df) // 4)
        df_prueba = df.iloc[inicio_prueba : inicio_prueba + tamaño_prueba]

        return df_prueba.reset_index(drop=True)

    def evaluar_en_prueba(self, nombre_dataset):
        """Evalúa el modelo en una partición de prueba local."""
        print(f"📊 Evaluando en dataset: {nombre_dataset}")

        datos_prueba = self._cargar_dataset_local(nombre_dataset)

        tokenizado = self.tokenizador(
            datos_prueba["text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        etiquetas = datos_prueba["label"].astype(int).to_numpy()
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo.to(dispositivo)

        input_ids = tokenizado["input_ids"].to(dispositivo)
        attention_mask = tokenizado["attention_mask"].to(dispositivo)

        with torch.no_grad():
            salidas = self.modelo(input_ids=input_ids, attention_mask=attention_mask)
            predicciones = salidas.logits.argmax(dim=-1).cpu().numpy()

        return np.array(predicciones), np.array(etiquetas)

    def generar_tabla_metricas(self, predicciones, etiquetas):
        """Genera tabla de métricas."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        return {
            "Exactitud": accuracy_score(etiquetas, predicciones),
            "Precisión": precision_score(etiquetas, predicciones, average="weighted", zero_division=0),
            "Recall": recall_score(etiquetas, predicciones, average="weighted", zero_division=0),
            "F1": f1_score(etiquetas, predicciones, average="weighted", zero_division=0),
        }

    def graficar_matriz_confusion(self, predicciones, etiquetas, nombre_archivo="matriz_confusion.png"):
        """Genera matriz de confusión."""
        matriz_confusion = confusion_matrix(etiquetas, predicciones)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matriz_confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=range(matriz_confusion.shape[1]),
            yticklabels=range(matriz_confusion.shape[0]),
        )
        plt.xlabel("Etiqueta Predicha")
        plt.ylabel("Etiqueta Real")
        plt.title("Matriz de Confusión")

        ruta_guardado = os.path.join(self.directorio_salida, nombre_archivo)
        plt.savefig(ruta_guardado, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📈 Matriz de confusión guardada: {ruta_guardado}")
        return matriz_confusion

    def generar_reporte(self, predicciones, etiquetas, nombre_dataset):
        """Genera reporte completo en JSON."""
        reporte = classification_report(etiquetas, predicciones, output_dict=True, zero_division=0)

        archivo_reporte = os.path.join(self.directorio_salida, f"reporte_{nombre_dataset}.json")
        with open(archivo_reporte, "w", encoding="utf-8") as archivo:
            json.dump(reporte, archivo, indent=2)

        print(f"📋 Reporte guardado: {archivo_reporte}")
        return reporte

    def ejecutar_evaluacion_completa(self, nombre_dataset):
        """Ejecuta evaluación completa."""
        self.cargar_modelo_y_config()
        predicciones, etiquetas = self.evaluar_en_prueba(nombre_dataset)

        metricas = self.generar_tabla_metricas(predicciones, etiquetas)
        print("\n📊 Métricas:")
        for metrica, valor in metricas.items():
            print(f"  {metrica}: {valor:.4f}")

        self.graficar_matriz_confusion(predicciones, etiquetas, f"matriz_confusion_{nombre_dataset}.png")
        reporte = self.generar_reporte(predicciones, etiquetas, nombre_dataset)

        return metricas, reporte


def main():
    analizador = argparse.ArgumentParser(description="Evaluación exhaustiva")
    analizador.add_argument("--model_dir", required=True, help="Directorio del modelo")
    analizador.add_argument(
        "--conjunto_datos",
        required=True,
        choices=["resume_screening", "campus_recruitment", "student_performance"],
        help="Dataset a evaluar",
    )
    analizador.add_argument("--output_dir", default="./results", help="Directorio de salida")
    analizador.add_argument("--data_dir", default="./data", help="Directorio con CSVs locales")

    argumentos = analizador.parse_args()

    evaluador = EvaluadorAgotador(
        argumentos.model_dir,
        argumentos.output_dir,
        argumentos.data_dir,
    )
    evaluador.ejecutar_evaluacion_completa(argumentos.conjunto_datos)


if __name__ == "__main__":
    main()
