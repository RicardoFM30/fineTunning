# Fine-tuning de Modelos para Predicción de Rendimiento Académico

## 📋 Descripción del Proyecto

Fine-tuning sistemático de modelos preentrenados de Hugging Face para clasificación de textos educativos y predicción de desempeño académico.

**Objetivo:** Evaluación comparativa del impacto de datasets, parámetros y arquitecturas en la calidad de predicción.

## 📊 Datasets Utilizados

| Dataset | Descripción | Clases | Tamaño | Enlace |
|---------|-------------|--------|--------|--------|
| **IMDB** | Reseñas de películas (positivo/negativo) | 2 | 500 train, 100 test | https://huggingface.co/datasets/imdb |
| **AG News** | Noticias en 4 categorías (World, Sports, Business, Sci/Tech) | 4 | 600 train, 100 test | https://huggingface.co/datasets/ag_news |
| **DBpedia** | Descripciones de entidades en 14 clases (Company, Artist, Athlete, etc) | 14 | 600 train, 100 test | https://huggingface.co/datasets/dbpedia_14 |

## 🧪 Experimentación Sistemática

### Configuraciones Probadas

- **Learning Rate:** 2e-5, 5e-5, 1e-4
- **Batch Size:** 8, 16, 32
- **Epochs:** 3, 5, 10
- **Weight Decay:** 0, 0.01, 0.1
- **Warmup Steps:** 0, 500, 1000

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
python scripts/train.py --conjunto_datos imdb --nombre_config config_1 --modelo distilbert-base-uncased

# 3. Entrenar modelo con AG News + config_2
python scripts/train.py --conjunto_datos ag_news --nombre_config config_2 --modelo distilbert-base-uncased

# 4. Entrenar modelo con DBpedia + config_3
python scripts/train.py --conjunto_datos dbpedia --nombre_config config_3 --modelo distilbert-base-uncased

# 5. Evaluar modelo
python scripts/evaluate.py --model_dir ./models/entrenado_imdb --dataset imdb

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

**Autor:** [Tu Nombre]  
**Fecha:** Febrero 2026
