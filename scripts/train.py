"""
Script modular de entrenamiento para fine-tuning.
Soporta múltiples datasets y configuraciones.
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def validar_csv_requerido(ruta_archivo):
    """Valida que el CSV exista y tenga contenido antes de entrenar."""
    if not os.path.exists(ruta_archivo) or os.path.getsize(ruta_archivo) == 0:
        raise FileNotFoundError(
            f"❌ No se encontró el archivo requerido: {ruta_archivo}\n"
            "   Descárgalo manualmente y colócalo en ./data con ese nombre."
        )


class EntrenadorFineTuning:
    """
    Clase para manejar fine-tuning modular de modelos preentrenados.
    
    Flujo:
    1. cargar_dataset() - Carga el dataset especificado
    2. tokenizar() - Tokeniza los textos
    3. entrenar() - Entrena el modelo con los parámetros especificados
    
    Attributes:
        configuracion: Diccionario con toda la configuración del proyecto
        nombre_dataset: Nombre del dataset (resume_screening, campus_recruitment, student_performance)
        nombre_modelo: Nombre del modelo HF (ej: distilbert-base-uncased)
        configuracion_entrenamiento: Dict con hiperparámetros
        dispositivo: Dispositivo computacional (cuda o cpu)
        directorio_salida: Ruta donde se guardan los modelos
    """
    
    def __init__(self, configuracion, nombre_dataset, nombre_modelo, configuracion_entrenamiento):
        """
        Inicializa el entrenador.
        
        Args:
            configuracion: Diccionario de configuración general (cargado de config.yaml)
            nombre_dataset: Nombre del dataset a usar (resume_screening, campus_recruitment, student_performance)
            nombre_modelo: Nombre del modelo HF (ej: distilbert-base-uncased)
            configuracion_entrenamiento: Diccionario con hiperparámetros:
                - tasa_aprendizaje: learning rate
                - tamaño_lote: batch size
                - épocas: número de epochs
                - pasos_calentamiento: warmup steps
                - decaimiento_peso: weight decay
        """
        print(f"\n{'='*80}")
        print(f"🏗️  INICIALIZANDO ENTRENADOR")
        print(f"{'='*80}")
        print(f"  Dataset: {nombre_dataset}")
        print(f"  Modelo: {nombre_modelo}")
        print(f"  Config: {configuracion_entrenamiento}")
        print(f"{'='*80}\n")
        
        self.configuracion = configuracion
        self.nombre_dataset = nombre_dataset
        self.nombre_modelo = nombre_modelo
        self.configuracion_entrenamiento = configuracion_entrenamiento
        self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_to_id = None
        print(f"✅ Dispositivo: {self.dispositivo}")
        
        # Crear directorio para este experimento
        nombre_experimento = f"{nombre_dataset}_{nombre_modelo.split('/')[-1]}_{int(datetime.now().timestamp())}"
        self.directorio_salida = os.path.join(configuracion["rutas"]["directorio_modelos"], nombre_experimento)
        os.makedirs(self.directorio_salida, exist_ok=True)
        print(f"✅ Directorio salida: {self.directorio_salida}\n")
        
    def cargar_dataset(self):
        """
        Carga el dataset especificado desde archivos CSV locales.
        
        Soporta:
        - resume_screening: CVs clasificados por tipo de profesional
        - campus_recruitment: Perfiles de estudiantes y colocación
        - student_performance: Desempeño académico de estudiantes
        
        El dataset se limita al tamaño especificado en config.yaml.
        
        Sets:
            self.conjunto_entrenamiento: Dataset de entrenamiento
            self.conjunto_prueba: Dataset de prueba
            self.num_etiquetas: Número de clases
        """
        print(f"📦 Cargando dataset: {self.nombre_dataset}")
        config_dataset = self.configuracion["conjuntos_datos"][self.nombre_dataset]
        
        import pandas as pd
        
        # Rutas de los datasets
        ruta_datos = self.configuracion["rutas"]["directorio_datos"]
        
        if self.nombre_dataset == "resume_screening":
            # Resume Screening: Clasificar CVs por tipo de profesional
            print(f"   Fuente: ./data/resume_screening.csv")
            print(f"   Tarea: Clasificar CV → Tipo de profesional (IT, Finance, HR, etc.)")
            
            ruta_csv = f"{ruta_datos}/resume_screening.csv"
            validar_csv_requerido(ruta_csv)
            df = pd.read_csv(ruta_csv)
            
            # Formato clásico: resume_text + Category
            if "resume_text" in df.columns and "Category" in df.columns:
                df = df.rename(columns={"resume_text": "text", "Category": "label"})
                df = df[["text", "label"]].dropna()
            # Formato alternativo detectado: Skills/Experience/... + Job Role
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
                    "❌ resume_screening.csv no tiene columnas compatibles.\n"
                    f"   Columnas encontradas: {list(df.columns)}\n"
                    "   Se esperaba ('resume_text','Category') o una columna 'Job Role'."
                )
            
            # Convertir labels a índices numéricos
            labels_unicos = df["label"].unique()
            label_to_id = {label: idx for idx, label in enumerate(labels_unicos)}
            df["label"] = df["label"].map(label_to_id)
            
            # Dividir en train/test
            tamaño_entrenamiento = min(config_dataset["tamaño_entrenamiento"], len(df) // 2)
            tamaño_prueba = min(config_dataset["tamaño_prueba"], len(df) // 4)
            
            conjunto_entrenamiento = Dataset.from_pandas(df.head(tamaño_entrenamiento))
            conjunto_prueba = Dataset.from_pandas(df.iloc[len(df)//2:len(df)//2 + tamaño_prueba])
            
            num_etiquetas = len(labels_unicos)
            self.label_to_id = {str(k): int(v) for k, v in label_to_id.items()}
            
        elif self.nombre_dataset == "campus_recruitment":
            # Campus Recruitment: Clasificar por estado de colocación
            print(f"   Fuente: ./data/campus_recruitment.csv")
            print(f"   Tarea: Predecir colocación de estudiante (Colocado/No colocado)")
            
            ruta_csv = f"{ruta_datos}/campus_recruitment.csv"
            validar_csv_requerido(ruta_csv)
            df = pd.read_csv(ruta_csv)
            
            # Normalizaciones de columnas para variantes del dataset
            if "specialisation" in df.columns and "specialization" not in df.columns:
                df = df.rename(columns={"specialisation": "specialization"})
            if "degree_t" in df.columns and "Degree" not in df.columns:
                df = df.rename(columns={"degree_t": "Degree"})

            # Crear un "texto" sintetizado a partir de características
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
            
            # Columna de etiqueta
            if "status" in df.columns:
                df = df.rename(columns={"status": "label"})
            
            # Convertir labels a índices
            label_values = df["label"].unique()
            label_to_id = {label: idx for idx, label in enumerate(label_values) if pd.notna(label)}
            df["label"] = df["label"].map(label_to_id)
            df = df.dropna(subset=["label"])
            
            # Dividir
            tamaño_entrenamiento = min(config_dataset["tamaño_entrenamiento"], len(df) // 2)
            tamaño_prueba = min(config_dataset["tamaño_prueba"], len(df) // 4)
            
            conjunto_entrenamiento = Dataset.from_pandas(df.head(tamaño_entrenamiento)[["text", "label"]])
            conjunto_prueba = Dataset.from_pandas(df.iloc[len(df)//2:len(df)//2 + tamaño_prueba][["text", "label"]])
            
            num_etiquetas = len(label_to_id)
            self.label_to_id = {str(k): int(v) for k, v in label_to_id.items()}
            
        elif self.nombre_dataset == "student_performance":
            # Student Performance: Clasificar por desempeño
            print(f"   Fuente: ./data/student_performance.csv")
            print(f"   Tarea: Clasificar nivel de rendimiento académico")
            
            ruta_csv = f"{ruta_datos}/student_performance.csv"
            validar_csv_requerido(ruta_csv)
            df = pd.read_csv(ruta_csv)
            
            # Formato 1: students-performance-in-exams (math/reading/writing)
            if "math score" in df.columns and "reading score" in df.columns and "writing score" in df.columns:
                def crear_texto_academico(row):
                    textos = []
                    for col in ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]:
                        if col in row.index and pd.notna(row[col]):
                            textos.append(f"{col}: {row[col]}")
                    return " ".join(textos) if textos else "Estudiante"

                df["text"] = df.apply(crear_texto_academico, axis=1)
                df["puntuacion_promedio"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3
            # Formato 2: student-performance-data-set (G1/G2/G3)
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
                    "❌ student_performance.csv no tiene columnas compatibles.\n"
                    f"   Columnas encontradas: {list(df.columns)}\n"
                    "   Se esperaba (math score/reading score/writing score) o (G1/G2/G3)."
                )

            df["label"] = pd.cut(df["puntuacion_promedio"], bins=3, labels=["Bajo", "Medio", "Alto"], include_lowest=True)
            label_to_id = {"Bajo": 0, "Medio": 1, "Alto": 2}
            df["label"] = df["label"].astype(str).map(label_to_id)
            
            df = df.dropna(subset=["label"])
            
            # Dividir
            tamaño_entrenamiento = min(config_dataset["tamaño_entrenamiento"], len(df) // 2)
            tamaño_prueba = min(config_dataset["tamaño_prueba"], len(df) // 4)
            
            conjunto_entrenamiento = Dataset.from_pandas(df.head(tamaño_entrenamiento)[["text", "label"]])
            conjunto_prueba = Dataset.from_pandas(df.iloc[len(df)//2:len(df)//2 + tamaño_prueba][["text", "label"]])
            
            num_etiquetas = 3  # Bajo, Medio, Alto
            self.label_to_id = {"Bajo": 0, "Medio": 1, "Alto": 2}
            
        else:
            raise ValueError(f"❌ Dataset {self.nombre_dataset} no soportado. Disponibles: resume_screening, campus_recruitment, student_performance")
        
        self.conjunto_entrenamiento = conjunto_entrenamiento
        self.conjunto_prueba = conjunto_prueba
        self.num_etiquetas = num_etiquetas
        
        print(f"✅ Dataset cargado: {config_dataset['nombre']}")
        print(f"   Muestras entrenamiento: {len(conjunto_entrenamiento)}")
        print(f"   Muestras prueba: {len(conjunto_prueba)}")
        print(f"   Clases: {self.num_etiquetas}\n")
        
        return self.conjunto_entrenamiento, self.conjunto_prueba
    
    def tokenizar(self):
        """
        Tokeniza los textos de los datasets usando el tokenizador del modelo.
        
        Proceso:
        1. Carga el tokenizador preentrenado
        2. Define función de tokenización con:
           - padding: rellena a longitud máxima
           - truncation: recorta textos largos
           - max_length: 256 tokens
        3. Aplica tokenización a datasets entrenamiento y prueba
        4. Elimina columnas de texto (ya no necesarias)
        
        Sets:
            self.tokenizador: Tokenizador del modelo
            self.tokenizado_entrenamiento: Dataset tokenizado (train)
            self.tokenizado_prueba: Dataset tokenizado (test)
        """
        print(f"🔤 Tokenizando datasets con {self.nombre_modelo}...")
        
        self.tokenizador = AutoTokenizer.from_pretrained(self.nombre_modelo)
        print(f"✅ Tokenizador cargado\n")
        
        def funcion_tokenizar(ejemplos):
            """Tokeniza un batch de ejemplos."""
            return self.tokenizador(
                ejemplos["text"],
                padding="max_length",
                truncation=True,
                max_length=256
            )
        
        print(f"🔄 Tokenizando entrenamiento ({len(self.conjunto_entrenamiento)} ejemplos)...")
        self.tokenizado_entrenamiento = self.conjunto_entrenamiento.map(funcion_tokenizar, batched=True)
        
        print(f"🔄 Tokenizando prueba ({len(self.conjunto_prueba)} ejemplos)...")
        self.tokenizado_prueba = self.conjunto_prueba.map(funcion_tokenizar, batched=True)
        
        # Remover columnas de texto (ya no necesarias)
        columna_a_eliminar = "text" if "text" in self.tokenizado_entrenamiento.column_names else ("title" if "title" in self.tokenizado_entrenamiento.column_names else "content")
        self.tokenizado_entrenamiento = self.tokenizado_entrenamiento.remove_columns([columna_a_eliminar])
        self.tokenizado_prueba = self.tokenizado_prueba.remove_columns([columna_a_eliminar])
        
        print(f"✅ Tokenización completada")
        print(f"   Columnas finales: {self.tokenizado_entrenamiento.column_names}\n")
        
        return self.tokenizado_entrenamiento, self.tokenizado_prueba
    
    def calcular_metricas(self, predicciones_evaluacion):
        """
        Calcula métricas de evaluación completas durante el entrenamiento.
        
        Se ejecuta después de cada epoch de validación.
        
        Métricas calculadas:
        - Exactitud: % de predicciones correctas
        - F1 (weighted): Balance entre precisión y recall
        - Precisión (weighted): % de positivos predichos correctos
        - Recall (weighted): % de positivos reales detectados
        
        Args:
            predicciones_evaluacion: Tupla (logits, etiquetas)
                - logits: Salida bruta del modelo
                - etiquetas: Etiquetas verdaderas
        
        Returns:
            dict: Diccionario con todas las métricas calculadas
        """
        predicciones, etiquetas = predicciones_evaluacion
        predicciones = np.argmax(predicciones, axis=1)  # Convierte logits a clases
        
        # Usar sklearn.metrics para calcular métricas (más eficiente)
        metricas = {
            "accuracy": accuracy_score(etiquetas, predicciones),
            "f1": f1_score(etiquetas, predicciones, average="weighted", zero_division=0),
            "precision": precision_score(etiquetas, predicciones, average="weighted", zero_division=0),
            "recall": recall_score(etiquetas, predicciones, average="weighted", zero_division=0),
        }
        return metricas
    
    def entrenar(self):
        """
        Entrena el modelo preentrenado en el dataset especificado.
        
        Pasos:
        1. Carga el modelo preentrenado
        2. Configura los argumentos de entrenamiento
        3. Crea el Trainer de Hugging Face
        4. Ejecuta el entrenamiento
        5. Guarda el modelo y la configuración
        
        Returns:
            tuple: (entrenador, resultado_entrenamiento)
        """
        print(f"\n{'='*80}")
        print(f"🚀 INICIANDO ENTRENAMIENTO")
        print(f"{'='*80}")
        print(f"Configuración:")
        for clave, valor in self.configuracion_entrenamiento.items():
            print(f"  - {clave}: {valor}")
        print(f"{'='*80}\n")
        
        # Cargar modelo
        print(f"📥 Cargando modelo: {self.nombre_modelo}...")
        modelo = AutoModelForSequenceClassification.from_pretrained(
            self.nombre_modelo,
            num_labels=self.num_etiquetas
        ).to(self.dispositivo)
        print(f"✅ Modelo cargado en {self.dispositivo}\n")
        
        # Convertir strings a números si es necesario (por si vienen de YAML)
        tasa_aprendizaje = float(self.configuracion_entrenamiento["tasa_aprendizaje"])
        tamaño_lote = int(self.configuracion_entrenamiento["tamaño_lote"])
        épocas = int(self.configuracion_entrenamiento["épocas"])
        pasos_calentamiento = int(self.configuracion_entrenamiento.get("pasos_calentamiento", 500))
        decaimiento_peso = float(self.configuracion_entrenamiento.get("decaimiento_peso", 0.01))
        
        print(f"📊 Hiperparámetros procesados:")
        print(f"  - Tasa aprendizaje: {tasa_aprendizaje}")
        print(f"  - Tamaño lote: {tamaño_lote}")
        print(f"  - Épocas: {épocas}")
        print(f"  - Pasos calentamiento: {pasos_calentamiento}")
        print(f"  - Decaimiento peso: {decaimiento_peso}\n")
        
        # Configurar argumentos de entrenamiento
        print(f"⚙️  Configurando argumentos de entrenamiento...")
        argumentos_entrenamiento = TrainingArguments(
            output_dir=self.directorio_salida,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=tasa_aprendizaje,
            per_device_train_batch_size=tamaño_lote,
            per_device_eval_batch_size=tamaño_lote,
            num_train_epochs=épocas,
            weight_decay=decaimiento_peso,
            warmup_steps=pasos_calentamiento,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            seed=42,
            logging_steps=10,
            save_total_limit=2,
        )
        print(f"✅ Argumentos configurados\n")
        
        # Crear entrenador
        print(f"🤖 Creando Trainer...")
        entrenador = Trainer(
            model=modelo,
            args=argumentos_entrenamiento,
            train_dataset=self.tokenizado_entrenamiento,
            eval_dataset=self.tokenizado_prueba,
            compute_metrics=self.calcular_metricas,
        )
        print(f"✅ Trainer creado\n")
        
        # Entrenar
        print(f"\n{'='*80}")
        print(f"⏳ ENTRENANDO MODELO...")
        print(f"{'='*80}\n")
        resultado_entrenamiento = entrenador.train()
        
        # Guardar
        print(f"\n{'='*80}")
        print(f"💾 GUARDANDO MODELO")
        print(f"{'='*80}")
        entrenador.save_model(self.directorio_salida)
        self.tokenizador.save_pretrained(self.directorio_salida)
        print(f"✅ Modelo guardado en: {self.directorio_salida}")
        
        # Guardar configuración de entrenamiento
        archivo_config = os.path.join(self.directorio_salida, "training_config.json")
        with open(archivo_config, "w") as archivo:
            json.dump({
                "dataset": self.nombre_dataset,
                "model": self.nombre_modelo,
                "label_to_id": self.label_to_id,
                "training_config": self.configuracion_entrenamiento,
                "results": resultado_entrenamiento.metrics,
            }, archivo, indent=2)
        print(f"✅ Configuración guardada\n")
        
        print(f"{'='*80}")
        print(f"✨ ENTRENAMIENTO COMPLETADO")
        print(f"{'='*80}\n")
        
        return entrenador, resultado_entrenamiento


def main():
    """
    Función principal: parsea argumentos y ejecuta el flujo completo de entrenamiento.
    
    Uso:
        python scripts/train.py \\
            --conjunto_datos resume_screening \\
            --nombre_config config_1 \\
            --modelo distilbert-base-uncased
    
    Argumentos:
        --archivo_config: Ruta al config.yaml (default: config.yaml)
        --conjunto_datos: Dataset a usar (resume_screening, campus_recruitment, student_performance) - REQUERIDO
        --modelo: Modelo HF (default: distilbert-base-uncased)
        --nombre_config: Nombre de configuración en config.yaml (config_1, config_2...) - REQUERIDO
    
    Flujo:
        1. Parsea argumentos de línea de comandos
        2. Carga configuración desde YAML
        3. Extrae configuración de hiperparámetros
        4. Crea instancia de EntrenadorFineTuning
        5. Ejecuta: cargar_dataset() → tokenizar() → entrenar()
    """
    analizador = argparse.ArgumentParser(
        description="Fine-tuning de modelos preentrenados para análisis de talento estudiantil",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Resume Screening con configuración 1
  python scripts/train.py --conjunto_datos resume_screening --nombre_config config_1
  
  # Campus Recruitment con configuración 2
  python scripts/train.py --conjunto_datos campus_recruitment --nombre_config config_2
  
  # Student Performance con BERT en lugar de DistilBERT
  python scripts/train.py --conjunto_datos student_performance --nombre_config config_3 --modelo bert-base-uncased
        """
    )
    analizador.add_argument(
        "--archivo_config",
        default="config.yaml",
        help="Ruta al archivo de configuración YAML"
    )
    analizador.add_argument(
        "--conjunto_datos",
        required=True,
        choices=["resume_screening", "campus_recruitment", "student_performance"],
        help="Dataset a usar para el entrenamiento"
    )
    analizador.add_argument(
        "--modelo",
        default="distilbert-base-uncased",
        help="Modelo preentrenado de Hugging Face"
    )
    analizador.add_argument(
        "--nombre_config",
        required=True,
        help="Nombre de la configuración en config.yaml (ej: config_1, config_2)"
    )
    
    argumentos = analizador.parse_args()
    
    # Cargar configuración
    print(f"\n📂 Cargando configuración desde: {argumentos.archivo_config}")
    import yaml
    with open(argumentos.archivo_config, "r", encoding="utf-8") as archivo:
        configuracion = yaml.safe_load(archivo)
    print(f"✅ Configuración cargada\n")
    
    # Verificar que existe la configuración especificada
    if argumentos.nombre_config not in configuracion["configuraciones_entrenamiento"]:
        raise ValueError(
            f"❌ Configuración '{argumentos.nombre_config}' no encontrada.\n"
            f"   Disponibles: {list(configuracion['configuraciones_entrenamiento'].keys())}"
        )
    
    configuracion_entrenamiento = configuracion["configuraciones_entrenamiento"][argumentos.nombre_config]
    
    # Crear y ejecutar entrenador
    objeto_entrenador = EntrenadorFineTuning(
        configuracion,
        argumentos.conjunto_datos,
        argumentos.modelo,
        configuracion_entrenamiento
    )
    objeto_entrenador.cargar_dataset()
    objeto_entrenador.tokenizar()
    objeto_entrenador.entrenar()


if __name__ == "__main__":
    main()
