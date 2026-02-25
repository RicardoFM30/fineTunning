"""
Script de análisis comparativo: compara resultados entre configs, datasets y modelos
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class AnálisisComparativo:
    """Análisis comparativo de múltiples entrenamientos."""
    
    def __init__(self, directorio_modelos, directorio_salida="./results"):
        """
        Args:
            directorio_modelos: Directorio conteniendo modelos entrenados
            directorio_salida: Directorio para guardar análisis
        """
        self.directorio_modelos = directorio_modelos
        self.directorio_salida = directorio_salida
        os.makedirs(directorio_salida, exist_ok=True)
        
    def recopilar_resultados(self):
        """Recopila resultados de todos los entrenamientos."""
        resultados = []
        
        for carpeta_modelo in os.listdir(self.directorio_modelos):
            ruta_modelo = os.path.join(self.directorio_modelos, carpeta_modelo)
            if not os.path.isdir(ruta_modelo):
                continue
            
            archivo_config = os.path.join(ruta_modelo, "training_config.json")
            if not os.path.exists(archivo_config):
                continue
            
            with open(archivo_config, "r") as f:
                config = json.load(f)
            
            # Extraer información
            datos_resultado = {
                "id_experimento": carpeta_modelo,
                "dataset": config.get("dataset", "desconocido"),
                "modelo": config.get("model", "desconocido").split("/")[-1],
                "tasa_aprendizaje": config.get("training_config", {}).get("tasa_aprendizaje", "desconocido"),
                "tamaño_lote": config.get("training_config", {}).get("tamaño_lote", "desconocido"),
                "épocas": config.get("training_config", {}).get("épocas", "desconocido"),
            }
            
            # Agregar métricas si existen
            if "results" in config:
                for clave, valor in config["results"].items():
                    if isinstance(valor, (int, float)):
                        datos_resultado[clave] = valor
            
            resultados.append(datos_resultado)
        
        self.df_resultados = pd.DataFrame(resultados)
        print(f"📊 Se encontraron {len(resultados)} entrenamientos")
        return self.df_resultados
    
    def comparar_configuraciones(self):
        """Compara diferentes configuraciones para el mismo dataset/modelo."""
        if self.df_resultados.empty:
            print("⚠️ Sin resultados para analizar")
            return
        
        # Agrupar por dataset y modelo
        agrupado = self.df_resultados.groupby(["dataset", "modelo"])
        
        for (dataset, modelo), grupo in agrupado:
            print(f"\n📈 Comparativa: Dataset={dataset}, Modelo={modelo}")
            print(grupo[["tasa_aprendizaje", "tamaño_lote", "épocas", "eval_loss" if "eval_loss" in grupo.columns else "pérdida_evaluación"]].to_string())
    
    def generar_tabla_comparativa(self):
        """Genera tabla comparativa CSV."""
        if self.df_resultados.empty:
            return
        
        # Seleccionar columnas principales
        columnas_a_mantener = [col for col in self.df_resultados.columns if col not in ["id_experimento", "eval_loss"]]
        df_comparativa = self.df_resultados[columnas_a_mantener]
        
        # Guardar CSV
        archivo_csv = os.path.join(self.directorio_salida, "resultados_comparativos.csv")
        df_comparativa.to_csv(archivo_csv, index=False)
        print(f"💾 Tabla comparativa guardada: {archivo_csv}")
        
        return df_comparativa
    
    def graficar_impacto_dataset(self):
        """Visualiza impacto del dataset en rendimiento."""
        if self.df_resultados.empty:
            return
        
        # Exactitud por dataset
        fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
        
        if "eval_accuracy" in self.df_resultados.columns:
            columna_exactitud = "eval_accuracy"
        elif "accuracy" in self.df_resultados.columns:
            columna_exactitud = "accuracy"
        else:
            print("⚠️ No hay columna de exactitud para graficar")
            return
        
        # Gráfica 1: Exactitud por dataset
        sns.boxplot(data=self.df_resultados, x="dataset", y=columna_exactitud, ax=ejes[0])
        ejes[0].set_title("Exactitud por Dataset")
        ejes[0].set_ylabel("Exactitud")
        
        # Gráfica 2: Exactitud por modelo
        sns.boxplot(data=self.df_resultados, x="modelo", y=columna_exactitud, ax=ejes[1])
        ejes[1].set_title("Exactitud por Modelo")
        ejes[1].set_ylabel("Exactitud")
        
        plt.tight_layout()
        archivo_fig = os.path.join(self.directorio_salida, "impacto_dataset_modelo.png")
        plt.savefig(archivo_fig, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"📊 Gráfica de impacto guardada: {archivo_fig}")
    
    def graficar_impacto_hiperparametros(self):
        """Visualiza impacto de hiperparámetros."""
        if self.df_resultados.empty:
            return
        
        if "eval_accuracy" in self.df_resultados.columns:
            columna_exactitud = "eval_accuracy"
        elif "accuracy" in self.df_resultados.columns:
            columna_exactitud = "accuracy"
        else:
            return
        
        fig, ejes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Impacto tasa de aprendizaje
        if "tasa_aprendizaje" in self.df_resultados.columns:
            self.df_resultados.groupby("tasa_aprendizaje")[columna_exactitud].mean().plot(ax=ejes[0], marker='o')
            ejes[0].set_title("Impacto: Tasa de Aprendizaje")
            ejes[0].set_ylabel("Exactitud Promedio")
        
        # Impacto tamaño de lote
        if "tamaño_lote" in self.df_resultados.columns:
            self.df_resultados.groupby("tamaño_lote")[columna_exactitud].mean().plot(ax=ejes[1], marker='o')
            ejes[1].set_title("Impacto: Tamaño de Lote")
            ejes[1].set_ylabel("Exactitud Promedio")
        
        # Impacto épocas
        if "épocas" in self.df_resultados.columns:
            self.df_resultados.groupby("épocas")[columna_exactitud].mean().plot(ax=ejes[2], marker='o')
            ejes[2].set_title("Impacto: Épocas")
            ejes[2].set_ylabel("Exactitud Promedio")
        
        plt.tight_layout()
        archivo_fig = os.path.join(self.directorio_salida, "impacto_hiperparametros.png")
        plt.savefig(archivo_fig, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"📊 Gráfica de hiperparámetros guardada: {archivo_fig}")
    
    def generar_reporte_resumen(self):
        """Genera reporte resumido."""
        if self.df_resultados.empty:
            return
        
        reporte = []
        reporte.append("=" * 80)
        reporte.append("REPORTE COMPARATIVO DE ENTRENAMIENTOS")
        reporte.append("=" * 80)
        reporte.append("")
        reporte.append(f"Total de experimentos: {len(self.df_resultados)}")
        reporte.append(f"Datasets únicos: {self.df_resultados['dataset'].nunique()}")
        reporte.append(f"Modelos únicos: {self.df_resultados['modelo'].nunique()}")
        reporte.append("")
        reporte.append("RESULTADOS PRINCIPALES:")
        reporte.append("-" * 80)
        
        if "eval_accuracy" in self.df_resultados.columns:
            columna_exactitud = "eval_accuracy"
        elif "accuracy" in self.df_resultados.columns:
            columna_exactitud = "accuracy"
        else:
            columna_exactitud = None
        
        if columna_exactitud:
            reporte.append(f"Mejor exactitud: {self.df_resultados[columna_exactitud].max():.4f}")
            reporte.append(f"Peor exactitud: {self.df_resultados[columna_exactitud].min():.4f}")
            reporte.append(f"Exactitud promedio: {self.df_resultados[columna_exactitud].mean():.4f}")
        
        texto_reporte = "\n".join(reporte)
        
        # Guardar reporte
        archivo_reporte = os.path.join(self.directorio_salida, "resumen_comparativo.txt")
        with open(archivo_reporte, "w") as f:
            f.write(texto_reporte)
        
        print("\n" + texto_reporte)
        print(f"\n💾 Reporte guardado: {archivo_reporte}")
    
    def ejecutar_analisis_completo(self):
        """Ejecuta análisis completo."""
        print("🔍 Iniciando análisis comparativo...")
        
        self.recopilar_resultados()
        self.comparar_configuraciones()
        self.generar_tabla_comparativa()
        self.graficar_impacto_dataset()
        self.graficar_impacto_hiperparametros()
        self.generar_reporte_resumen()
        
        print("\n✅ Análisis completado")


def main():
    analizador = argparse.ArgumentParser(description="Análisis comparativo")
    analizador.add_argument("--models_dir", default="./models", help="Directorio de modelos")
    analizador.add_argument("--output_dir", default="./results", help="Directorio de salida")
    
    argumentos = analizador.parse_args()
    
    analisis = AnálisisComparativo(argumentos.models_dir, argumentos.output_dir)
    analisis.ejecutar_analisis_completo()


if __name__ == "__main__":
    main()
