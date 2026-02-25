# Copilot Instructions - Fine-tuning Académico

## 🎯 Propósito del Proyecto

Fine-tuning de modelos preentrenados (Hugging Face Transformers) para clasificación de textos académicos. Objetivo: maximizar puntuación en rúbrica que valora **variedad de datasets (3+), experimentación sistemática (5+ configs), evaluación exhaustiva y análisis comparativo profundo**.

## 🏗️ Arquitectura del Proyecto

### Estructura de Carpetas
```
scripts/          → Scripts modulares de entrenamiento y evaluación
notebooks/        → Exploraciones y visualizaciones
data/            → Datasets (.csv, .json) - 3 datasets diferenciados
models/          → Modelos entrenados por configuración
results/         → Gráficas, métricas, análisis comparativos
config.yaml      → Hiperparámetros centralizados
```

### Flujo de Datos Principal
1. **Load Dataset** → CSVs locales de talento estudiantil (Kaggle)
2. **Tokenize** → `distilbert-base-uncased` tokenizer
3. **Train** → `Trainer` API con múltiples configs (5+)
4. **Evaluate** → Accuracy, F1, Precision, Recall, Confusion Matrix
5. **Compare** → Análisis de impacto dataset + parámetros
6. **Document** → Resultados en tablas y gráficas

## 🔑 Patrones Críticos

### 1. Datasets Diferenciados (Rúbrica: Nivel 5)
Usar 3 datasets reales con características distintas:
- **Resume Screening** (https://www.kaggle.com/datasets/mfaisalqureshi/resume-screening-dataset) - 25 clases
- **Campus Recruitment** (https://www.kaggle.com/datasets/benroshan/campus-recruitment-data) - 2 clases
- **Student Performance** (https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) - 3 clases

*Patrón:* Cada dataset se carga desde `./data/*.csv` y se normaliza en `train.py`/`data_utils.py`.

### 2. Experimentación Sistemática (Rúbrica: Nivel 5)
Mínimo 5 configuraciones variando múltiples parámetros:
```yaml
# En config.yaml, cada config_N define:
config_1: learning_rate=2e-5, batch_size=8, epochs=3
config_2: learning_rate=5e-5, batch_size=16, epochs=5
config_3: learning_rate=1e-4, batch_size=32, epochs=10
config_4: learning_rate=2e-5, batch_size=16, epochs=5, weight_decay=0
config_5: learning_rate=5e-5, batch_size=8, epochs=10, warmup_steps=1000, weight_decay=0.1
```
Con 3 datasets × 5 configs = **15 entrenamientos sistemáticos**.
*Patrón:* Loop sobre configs en script `train.py` que crea directorio `/models/config_N/` para cada una.

### 3. Evaluación Exhaustiva (Rúbrica: Nivel 5)
Para cada entrenamiento, computar:
- Accuracy, F1, Precision, Recall (via `evaluate` library)
- Matriz de confusión (via `sklearn.metrics.confusion_matrix`)
- Gráficas de pérdida (train/eval loss por epoch)
- Análisis overfitting: comparar train vs eval metrics

*Patrón:* Función `compute_metrics()` integrada en `TrainingArguments` + post-procesamiento en `evaluate.py`.

### 4. Análisis Comparativo Profundo (Rúbrica: Nivel 5)
Guardar resultados en tablas CSV/JSON:
```
results/
├── config_comparison.csv  # Accuracy, F1 de cada config
├── dataset_impact.csv     # Impacto del tamaño/tipo dataset
└── loss_curves/           # Gráficas train/eval loss
```

*Patrón:* Script `compare_results.py` que:
- Lee logs de todos los entrenamientos
- Crea tablas comparativas
- Genera gráficas (matplotlib/seaborn)
- Analiza correlaciones (dataset size vs accuracy)

## 📋 Developer Workflows

### 1. Agregar Nuevo Dataset (Real o Personalizado)
**Datasets reales (Kaggle + CSV local):**
```python
# En config.yaml dentro de conjuntos_datos:
nueva_dataset:
  nombre: "Nombre descriptivo"
  # Enlace: https://www.kaggle.com/datasets/...
  tamaño_entrenamiento: 600
  tamaño_prueba: 100
  num_etiquetas: 5
  descripcion: "Descripción..."

# En scripts/train.py, agregar caso en cargar_dataset():
elif self.nombre_dataset == "nueva_dataset":
    # Cargar CSV en ./data y normalizar columnas text/label
    ...
```

**Datasets personalizados (CSV local):**
```python
# Usar función en scripts/data_utils.py:
conjunto = cargar_dataset_talento_desde_csv("resume_screening")
```

### 2. Entrenar Nueva Configuración
```bash
python scripts/train.py \
  --archivo_config config.yaml \
  --conjunto_datos resume_screening \
  --modelo distilbert-base-uncased \
  --nombre_config config_1
```
*Patrón:* Argparse con fallback a `config.yaml` si no se especifica.

### 3. Evaluar y Comparar
```bash
python scripts/evaluate.py --model_dir ./models/<modelo_entrenado> --conjunto_datos resume_screening
python scripts/compare_results.py --output_dir ./results
```

## 🚫 Antipatrones a Evitar

❌ **No hacer:**
- Hardcodear paths absolutos (usar `config.yaml`)
- Usar single dataset sin variación
- Sacar solo accuracy sin F1/precisión/recall
- Entrenar con 1-2 configuraciones fijas
- Sin control de random seeds (usar `seed=42` consistentemente)

✅ **Hacer:**
- Todos los parámetros en `config.yaml`
- Loops sobre múltiples datasets y configs
- Métricas completas + visualización
- 5+ configuraciones sistemáticas
- Reproducibilidad con seeds

## 📦 Dependencias Clave

- `transformers==4.49.0` → Modelos HF, Trainer API
- `datasets==3.5.0` → Estructuras Dataset para entrenamiento
- `torch==2.6.0` → Backend compu
- `evaluate==0.4.3` → Calcular métricas
- `scikit-learn==1.6.1` → Matriz confusión, análisis
- `matplotlib/seaborn` → Gráficas

## 🎓 Contexto Académico

**Rúbrica Key Criteria:**
- L5 (Máxima) = 3+ datasets + 5+ configs + evaluación exhaustiva + análisis profundo
- L4 = 3 datasets + 4+ parámetros + matrices + overfitting
- L3 = 2 datasets + 3 configs + métricas + comparación

*Este proyecto apunta a Nivel 5 en todas categorías.*

## 💡 Sugerencias para AI Agents

Cuando agregues código:
1. **Propón** nuevas funciones siguiendo estructura de `scripts/`
2. **Valida** que use parámetros desde `config.yaml`
3. **Incluir** logging de ejecución en `/results/`
4. **Documentar** nuevos datasets agregados
5. **Proponer** gráficas comparativas para `/results/`

---

**Última actualización:** Febrero 2026
