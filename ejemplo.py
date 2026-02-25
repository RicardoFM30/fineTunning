from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
import evaluate
import numpy as np

# ===============================
# 1. CARGAR DATASET
# ===============================
dataset = load_dataset("imdb")
train_dataset = dataset["train"].shuffle(seed=21).select(range(500))
test_dataset = dataset["test"].shuffle(seed=21).select(range(100))

# ===============================
# 2. CARGAR TOKENIZER
# ===============================
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# ===============================
# 3. TOKENIZACIÓN
# ===============================
def tokenize_function(examples):
    """
    Tokeniza los ejemplos del dataset.
    - padding: obliga a que todas las palabras tengan la misma longitud (añade espacios vacíos)
    - truncation: recorta el número máximo de tokens de un texto
    - batched: procesa los datos en grupos (lotes) en lugar de uno por uno
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# ===============================
# 4. CARGAR MODELO
# ===============================
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2  # positivo/negativo
)

# ===============================
# 5. DEFINIR MÉTRICA
# ===============================
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# ===============================
# 6. CONFIGURAR ENTRENAMIENTO
# ===============================
training_args = TrainingArguments(
    output_dir="test_trainer",  # Ruta donde se guardará el entrenamiento
    evaluation_strategy="epoch",  # Valida cada epoch (cada pasada completa por el dataset)
    per_device_train_batch_size=2,  # Elementos procesados a la vez en entrenamiento
    per_device_eval_batch_size=2,  # Elementos procesados a la vez en evaluación
    learning_rate=2e-5,  # Ratio de aprendizaje
    num_train_epochs=1,  # Número de épocas
)

# ===============================
# 7. ENTRENAR MODELO
# ===============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)
trainer.train()

# ===============================
# 8. GUARDAR MODELO
# ===============================
model.save_pretrained("./mi_modelo_entrenado")
tokenizer.save_pretrained("./mi_modelo_entrenado")

# ===============================
# 9. USAR MODELO PARA PREDICCIÓN
# ===============================
clasificador = pipeline(
    "sentiment-analysis",
    model="./mi_modelo_entrenado",
    tokenizer="./mi_modelo_entrenado"
)

frase = "This movie was surprisingly good, I enjoyed it a lot!"
resultado = clasificador(frase)
print(resultado)
