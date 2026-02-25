"""
Ejemplo mínimo de fine-tuning para talento estudiantil/juvenil.
Usa el dataset local `resume_screening.csv` y entrena una corrida corta.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

from scripts.data_utils import cargar_dataset_talento_desde_csv


# 1) Cargar dataset local
conjunto, label_to_id = cargar_dataset_talento_desde_csv(
    "resume_screening",
    directorio_datos="./data",
    tamaño_entrenamiento=500,
    tamaño_prueba=100,
)
train_dataset = conjunto["train"]
test_dataset = conjunto["test"]

# 2) Tokenizador
modelo_base = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(modelo_base)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)


tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_test = tokenized_test.remove_columns(["text"])

# 3) Modelo
model = AutoModelForSequenceClassification.from_pretrained(
    modelo_base,
    num_labels=len(label_to_id),
)

# 4) Métrica
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# 5) Entrenamiento breve
training_args = TrainingArguments(
    output_dir="test_trainer_talento",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    num_train_epochs=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)
trainer.train()

# 6) Guardado
model.save_pretrained("./mi_modelo_talento")
tokenizer.save_pretrained("./mi_modelo_talento")
print("Modelo guardado en ./mi_modelo_talento")
