# 🎓 Fine-tuning Académico - Análisis de Talento de Estudiantes

## 📋 Requisitos Previos

- **Python:** 3.8+
- **Espacio:** ~3GB mínimo (datos + modelos)
- **Conexión internet:** Para descargar datasets de Kaggle (única vez)
- **Entorno virtual:** incluido (`hf-finetuning/`)

---

## 🚀 Comenzar: 4 Pasos

### 1. Instalar dependencias
```bash
# Activar entorno virtual (Windows)
hf-finetuning\Scripts\Activate.ps1

# Instalar paquetes (incluye kagglehub)
pip install -r requirements.txt
```

### 2. Descargar datasets de Kaggle
```bash
# Descarga datasets sobre talento de estudiantes (primera única vez)
python scripts/download_datasets.py

# Genera:
#   - data/resume_screening.csv (CVs clasificados)
#   - data/campus_recruitment.csv (Colocación en campus)
#   - data/student_performance.csv (Rendimiento académico)
```

### 3. Entrenar primer modelo (5 minutos)
```bash
python scripts/train.py --conjunto_datos resume_screening --nombre_config config_1
```

### 4. Entrenar 15 modelos para máxima puntuación en rúbrica

Ejecuta estos comandos para entrenar 3 datasets × 5 configuraciones:

**Resume Screening (Clasificación de profesionales):**
```bash
python scripts/train.py --conjunto_datos resume_screening --nombre_config config_1
python scripts/train.py --conjunto_datos resume_screening --nombre_config config_2
python scripts/train.py --conjunto_datos resume_screening --nombre_config config_3
python scripts/train.py --conjunto_datos resume_screening --nombre_config config_4
python scripts/train.py --conjunto_datos resume_screening --nombre_config config_5
```

**Campus Recruitment (Predicción de colocación):**
```bash
python scripts/train.py --conjunto_datos campus_recruitment --nombre_config config_1
python scripts/train.py --conjunto_datos campus_recruitment --nombre_config config_2
python scripts/train.py --conjunto_datos campus_recruitment --nombre_config config_3
python scripts/train.py --conjunto_datos campus_recruitment --nombre_config config_4
python scripts/train.py --conjunto_datos campus_recruitment --nombre_config config_5
```

**Student Performance (Rendimiento académico):**
```bash
python scripts/train.py --conjunto_datos student_performance --nombre_config config_1
python scripts/train.py --conjunto_datos student_performance --nombre_config config_2
python scripts/train.py --conjunto_datos student_performance --nombre_config config_3
python scripts/train.py --conjunto_datos student_performance --nombre_config config_4
python scripts/train.py --conjunto_datos student_performance --nombre_config config_5
```

**Análisis comparativo:**
```bash
python scripts/compare_results.py --models_dir ./models --output_dir ./results
```

---

## 📊 Datasets sobre Talento de Estudiantes

| Dataset | Tarea | Clases | Descripción |
|---------|-------|--------|-------------|
| **Resume Screening** | Clasificar CV por profesión | 25 | CVs de estudiantes con tipo de profesional (IT, Finance, HR, Engineering, etc.) |
| **Campus Recruitment** | Predicción de colocación | 2 | Perfiles estudiantiles: Colocado/No colocado en campus recruitment |
| **Student Performance** | Nivel de rendimiento | 3 | Rendimiento académico: Bajo/Medio/Alto basado en puntajes y características |

---

## ⚙️ Configuración (`config.yaml`) - CRÍTICO

**TODO está en `config.yaml`. NO edites código Python.**

Contiene:
- **5+ configuraciones:** learning_rate, batch_size, epochs, warmup_steps, weight_decay
- **3 datasets:** sobre talento de estudiantes (Resume, Recruitment, Performance)
- **Rutas:** directorio_datos, directorio_modelos, directorio_resultados

Ver [config.yaml](config.yaml) para valores específicos.

## 📈 Resultados

Ver carpeta `/results/` para gráficas comparativas, matrices de confusión y análisis detallado.

## 🏗️ Estructura del Proyecto

```
├── notebooks/           # Jupyter notebooks de exploración
├── scripts/            # Scripts de fine-tuning y evaluación
├── data/               # Datasets
├── models/             # Modelos entrenados
├── results/            # Resultados, gráficas, análisis
├── config.yaml         # Configuración centralizada
├── requirements.txt    # Dependencias
└── README.md          # Este archivo
```

## 🚀 Uso Rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar modelo con IMDB + config_1
python scripts/train.py --conjunto_datos imdb --nombre_config config_1

# 3. Entrenar modelo con AG News + config_2
python scripts/train.py --conjunto_datos ag_news --nombre_config config_2 --modelo bert-base-uncased

# 4. Entrenar modelo con DBpedia + config_3
python scripts/train.py --conjunto_datos dbpedia --nombre_config config_3

# 5. Evaluar modelo (reemplaza con tu ruta del modelo entrenado)
python scripts/evaluate.py --model_dir ./models/imdb_distilbert-base-uncased_** --conjunto_datos imdb

# 6. Análisis comparativo de todos los entrenamientos
python scripts/compare_results.py --models_dir ./models --output_dir ./results
```

## 📝 Tareas por Completar

- [ ] Implementar script de entrenamiento modular
- [ ] Crear notebook de exploración de datasets
- [ ] Configurar experimentación sistemática
- [ ] Generar gráficas comparativas
- [ ] Documentar conclusiones

---

**Autor:** Ricardo Fernandez Guzmán 
**Fecha:** 23 Febrero 2026
