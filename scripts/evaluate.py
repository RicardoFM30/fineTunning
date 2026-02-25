"""
Script de evaluación exhaustiva: métricas, gráficas, análisis comparativo
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset


class EvaluadorAgotador:
    """Evaluación exhaustiva de modelos entrenados."""
    
    def __init__(self, directorio_modelo, directorio_salida="./results"):
        """
        Args:
            directorio_modelo: Directorio del modelo entrenado
            directorio_salida: Directorio para guardar resultados
        """
        self.directorio_modelo = directorio_modelo
        self.directorio_salida = directorio_salida
        os.makedirs(directorio_salida, exist_ok=True)
        
        print(f"📂 Evaluando modelo: {directorio_modelo}")
        
    def cargar_modelo_y_config(self):
        """Carga el modelo y su configuración."""
        self.tokenizador = AutoTokenizer.from_pretrained(self.directorio_modelo)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(self.directorio_modelo)
        
        # Cargar configuración de entrenamiento
        archivo_config = os.path.join(self.directorio_modelo, "training_config.json")
        if os.path.exists(archivo_config):
            with open(archivo_config, "r") as f:
                self.configuracion = json.load(f)
        else:
            self.configuracion = {}
        
        print("✅ Modelo y configuración cargados")
        
    def evaluar_en_prueba(self, nombre_dataset):
        """Evalúa el modelo en el dataset de prueba."""
        print(f"📊 Evaluando en dataset: {nombre_dataset}")
        
        # Cargar dataset
        if nombre_dataset == "imdb":
            # Enlace: https://huggingface.co/datasets/imdb
            conjunto_datos = load_dataset("imdb")
            datos_prueba = conjunto_datos["test"].shuffle(seed=42).select(range(100))
        elif nombre_dataset == "ag_news":
            # Enlace: https://huggingface.co/datasets/ag_news
            conjunto_datos = load_dataset("ag_news")
            datos_prueba = conjunto_datos["test"].shuffle(seed=42).select(range(100))
        elif nombre_dataset == "dbpedia":
            # Enlace: https://huggingface.co/datasets/dbpedia_14
            # Dataset público: https://www.dbpedia.org/
            conjunto_datos = load_dataset("dbpedia_14")
            datos_prueba = conjunto_datos["test"].shuffle(seed=42).select(range(100))
        else:
            raise ValueError(f"Dataset {nombre_dataset} no soportado")
        
        # Tokenizar
        tokenizado = datos_prueba.map(
            lambda x: self.tokenizador(
                x.get("text", x.get("title", x.get("content", ""))),
                padding="max_length",
                truncation=True,
                max_length=256
            ),
            batched=True
        )
        
        # Inferencia
        predicciones = []
        etiquetas = datos_prueba["label"]
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo.to(dispositivo)
        
        for ejemplo in tokenizado:
            entradas = {k: torch.tensor([ejemplo[k]]).to(dispositivo) for k in ["input_ids", "attention_mask"]}
            with torch.no_grad():
                salidas = self.modelo(**entradas)
            prediccion = salidas.logits.argmax(dim=-1).item()
            predicciones.append(prediccion)
        
        return np.array(predicciones), np.array(etiquetas)
    
    def generar_tabla_metricas(self, predicciones, etiquetas):
        """Genera tabla de métricas."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        metricas = {
            "Exactitud": accuracy_score(etiquetas, predicciones),
            "Precisión": precision_score(etiquetas, predicciones, average="weighted", zero_division=0),
            "Recall": recall_score(etiquetas, predicciones, average="weighted", zero_division=0),
            "F1": f1_score(etiquetas, predicciones, average="weighted", zero_division=0),
        }
        
        return metricas
    
    def graficar_matriz_confusion(self, predicciones, etiquetas, nombre_archivo="matriz_confusion.png"):
        """Genera matriz de confusión."""
        matriz_confusión = confusion_matrix(etiquetas, predicciones)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(matriz_confusión, annot=True, fmt="d", cmap="Blues", xticklabels=range(matriz_confusión.shape[0]), yticklabels=range(matriz_confusión.shape[1]))
        plt.xlabel("Etiqueta Predicha")
        plt.ylabel("Etiqueta Real")
        plt.title("Matriz de Confusión")
        
        ruta_guardado = os.path.join(self.directorio_salida, nombre_archivo)
        plt.savefig(ruta_guardado, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"📈 Matriz de confusión guardada: {ruta_guardado}")
        return matriz_confusión
    
    def generar_reporte(self, predicciones, etiquetas, nombre_dataset):
        """Genera reporte completo."""
        reporte = classification_report(etiquetas, predicciones, output_dict=True)
        
        # Guardar como JSON
        archivo_reporte = os.path.join(self.directorio_salida, f"reporte_{nombre_dataset}.json")
        with open(archivo_reporte, "w") as f:
            json.dump(reporte, f, indent=2)
        
        print(f"📋 Reporte guardado: {archivo_reporte}")
        return reporte
    
    def ejecutar_evaluacion_completa(self, nombre_dataset):
        """Ejecuta evaluación completa."""
        self.cargar_modelo_y_config()
        
        # Evaluar
        predicciones, etiquetas = self.evaluar_en_prueba(nombre_dataset)
        
        # Métricas
        metricas = self.generar_tabla_metricas(predicciones, etiquetas)
        print("\n📊 Métricas:")
        for metrica, valor in metricas.items():
            print(f"  {metrica}: {valor:.4f}")
        
        # Gráficas
        self.graficar_matriz_confusion(predicciones, etiquetas, f"matriz_confusion_{nombre_dataset}.png")
        
        # Reporte detallado
        reporte = self.generar_reporte(predicciones, etiquetas, nombre_dataset)
        
        return metricas, reporte


def main():
    analizador = argparse.ArgumentParser(description="Evaluación exhaustiva")
    analizador.add_argument("--model_dir", required=True, help="Directorio del modelo")
    analizador.add_argument("--conjunto_datos", required=True, help="Dataset a evaluar (imdb, ag_news, dbpedia)")
    analizador.add_argument("--output_dir", default="./results", help="Directorio de salida")
    
    argumentos = analizador.parse_args()
    
    evaluador = EvaluadorAgotador(argumentos.model_dir, argumentos.output_dir)
    metricas, reporte = evaluador.ejecutar_evaluacion_completa(argumentos.conjunto_datos)


if __name__ == "__main__":
    main()
