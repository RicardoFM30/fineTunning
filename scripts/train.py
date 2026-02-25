"""
Script modular de entrenamiento para fine-tuning.
Soporta múltiples datasets y configuraciones.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate


class EntrenadorFineTuning:
    """Clase para manejar fine-tuning modular."""
    
    def __init__(self, configuracion, nombre_dataset, nombre_modelo, configuracion_entrenamiento):
        """
        Args:
            configuracion: Diccionario de configuración general
            nombre_dataset: Nombre del dataset a usar
            nombre_modelo: Nombre del modelo HF (ej: distilbert-base-uncased)
            configuracion_entrenamiento: Diccionario con hiperparámetros (lr, epochs, batch_size, etc)
        """
        self.configuracion = configuracion
        self.nombre_dataset = nombre_dataset
        self.nombre_modelo = nombre_modelo
        self.configuracion_entrenamiento = configuracion_entrenamiento
        self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Crear directorio para este experimento
        nombre_experimento = f"{nombre_dataset}_{nombre_modelo.split('/')[-1]}_{int(datetime.now().timestamp())}"
        self.directorio_salida = os.path.join(configuracion["paths"]["models_dir"], nombre_experimento)
        os.makedirs(self.directorio_salida, exist_ok=True)
        
    def cargar_dataset(self):
        """Carga el dataset especificado."""
        print(f"📦 Cargando dataset: {self.nombre_dataset}")
        
        config_dataset = self.configuracion["conjuntos_datos"][self.nombre_dataset]
        
        if self.nombre_dataset == "imdb":
            # Enlace: https://huggingface.co/datasets/imdb
            conjunto_datos = load_dataset("imdb")
            conjunto_entrenamiento = conjunto_datos["train"].shuffle(seed=42).select(range(config_dataset["tamaño_entrenamiento"]))
            conjunto_prueba = conjunto_datos["test"].shuffle(seed=42).select(range(config_dataset["tamaño_prueba"]))
            
        elif self.nombre_dataset == "ag_news":
            # Enlace: https://huggingface.co/datasets/ag_news
            conjunto_datos = load_dataset("ag_news")
            conjunto_entrenamiento = conjunto_datos["train"].shuffle(seed=42).select(range(config_dataset["tamaño_entrenamiento"]))
            conjunto_prueba = conjunto_datos["test"].shuffle(seed=42).select(range(config_dataset["tamaño_prueba"]))
            
        elif self.nombre_dataset == "dbpedia":
            # Enlace: https://huggingface.co/datasets/dbpedia_14
            # Dataset público: https://www.dbpedia.org/
            conjunto_datos = load_dataset("dbpedia_14")
            conjunto_entrenamiento = conjunto_datos["train"].shuffle(seed=42).select(range(config_dataset["tamaño_entrenamiento"]))
            conjunto_prueba = conjunto_datos["test"].shuffle(seed=42).select(range(config_dataset["tamaño_prueba"]))
            
        else:
            raise ValueError(f"Dataset {self.nombre_dataset} no soportado")
        
        self.conjunto_entrenamiento = conjunto_entrenamiento
        self.conjunto_prueba = conjunto_prueba
        self.num_etiquetas = config_dataset["num_etiquetas"]
        print(f"✅ Dataset cargado: {config_dataset['nombre']}")
        return self.conjunto_entrenamiento, self.conjunto_prueba
    
    def tokenizar(self):
        """Tokeniza los datasets."""
        print(f"🔤 Tokenizando con {self.nombre_modelo}...")
        
        tokenizador = AutoTokenizer.from_pretrained(self.nombre_modelo)
        
        def funcion_tokenizar(ejemplos):
            # El campo de texto varía según dataset
            campo_texto = "text" if "text" in ejemplos else "title"
            return tokenizador(
                ejemplos[campo_texto],
                padding="max_length",
                truncation=True,
                max_length=256
            )
        
        tokenizado_entrenamiento = self.conjunto_entrenamiento.map(funcion_tokenizar, batched=True)
        tokenizado_prueba = self.conjunto_prueba.map(funcion_tokenizar, batched=True)
        
        # Remover columnas no necesarias
        tokenizado_entrenamiento = tokenizado_entrenamiento.remove_columns(["text" if "text" in tokenizado_entrenamiento.column_names else "title"])
        tokenizado_prueba = tokenizado_prueba.remove_columns(["text" if "text" in tokenizado_prueba.column_names else "title"])
        
        self.tokenizador = tokenizador
        self.tokenizado_entrenamiento = tokenizado_entrenamiento
        self.tokenizado_prueba = tokenizado_prueba
        print("✅ Tokenización completada")
        return tokenizado_entrenamiento, tokenizado_prueba
    
    def calcular_metricas(self, predicciones_evaluacion):
        """Calcula métricas de evaluación."""
        predicciones, etiquetas = predicciones_evaluacion
        predicciones = np.argmax(predicciones, axis=1)
        
        # Cargar todas las métricas necesarias
        exactitud = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")
        
        metricas = {
            "accuracy": exactitud.compute(predictions=predicciones, references=etiquetas)["accuracy"],
            "f1": f1.compute(predictions=predicciones, references=etiquetas, average="weighted")["f1"],
            "precision": precision.compute(predictions=predicciones, references=etiquetas, average="weighted")["precision"],
            "recall": recall.compute(predictions=predicciones, references=etiquetas, average="weighted")["recall"],
        }
        return metricas
    
    def entrenar(self):
        """Entrena el modelo."""
        print(f"🚀 Iniciando entrenamiento con config: {self.configuracion_entrenamiento}")
        
        # Cargar modelo
        modelo = AutoModelForSequenceClassification.from_pretrained(
            self.nombre_modelo,
            num_labels=self.num_etiquetas
        ).to(self.dispositivo)
        
        # Configurar argumentos de entrenamiento
        argumentos_entrenamiento = TrainingArguments(
            output_dir=self.directorio_salida,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.configuracion_entrenamiento["learning_rate"],
            per_device_train_batch_size=self.configuracion_entrenamiento["batch_size"],
            per_device_eval_batch_size=self.configuracion_entrenamiento["batch_size"],
            num_train_epochs=self.configuracion_entrenamiento["epochs"],
            weight_decay=self.configuracion_entrenamiento["weight_decay"],
            warmup_steps=self.configuracion_entrenamiento.get("warmup_steps", 500),
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            seed=42,
        )
        
        # Crear entrenador
        entrenador = Trainer(
            model=modelo,
            args=argumentos_entrenamiento,
            train_dataset=self.tokenizado_entrenamiento,
            eval_dataset=self.tokenizado_prueba,
            compute_metrics=self.calcular_metricas,
        )
        
        # Entrenar
        resultado_entrenamiento = entrenador.train()
        entrenador.save_model(self.directorio_salida)
        self.tokenizador.save_pretrained(self.directorio_salida)
        
        # Guardar configuración
        archivo_config = os.path.join(self.directorio_salida, "training_config.json")
        with open(archivo_config, "w") as archivo:
            json.dump({
                "dataset": self.nombre_dataset,
                "model": self.nombre_modelo,
                "training_config": self.configuracion_entrenamiento,
                "results": resultado_entrenamiento.metrics,
            }, archivo, indent=2)
        
        print(f"✅ Modelo guardado en: {self.directorio_salida}")
        return entrenador, resultado_entrenamiento


def main():
    analizador = argparse.ArgumentParser(description="Fine-tuning modular")
    analizador.add_argument("--archivo_config", default="config.yaml", help="Ruta al archivo de configuración")
    analizador.add_argument("--conjunto_datos", required=True, help="Nombre del dataset (imdb, ag_news, dbpedia)")
    analizador.add_argument("--modelo", default="distilbert-base-uncased", help="Modelo HF")
    analizador.add_argument("--nombre_config", required=True, help="Nombre de configuración (config_1, config_2, etc)")
    
    argumentos = analizador.parse_args()
    
    # Cargar configuración
    import yaml
    with open(argumentos.archivo_config, "r") as archivo:
        configuracion = yaml.safe_load(archivo)
    
    configuracion_entrenamiento = configuracion["configuraciones_entrenamiento"][argumentos.nombre_config]
    
    # Crear y ejecutar entrenador
    objeto_entrenador = EntrenadorFineTuning(configuracion, argumentos.conjunto_datos, argumentos.modelo, configuracion_entrenamiento)
    objeto_entrenador.cargar_dataset()
    objeto_entrenador.tokenizar()
    objeto_entrenador.entrenar()


if __name__ == "__main__":
    main()
