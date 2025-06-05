import os
import argparse
from datasets import load_from_disk
from sklearn.metrics import classification_report
from google.cloud import storage
import numpy as np
from contextlib import contextmanager # Cambios: 2
import tqdm # Cambios: 2
from transformers import (
    AutoTokenizer, 
    Trainer, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding, 
    TrainingArguments
)

def download_from_gcs(bucket_name, prefix, destination_dir):
    print(f"🔽 Descargando de gs://{bucket_name}/{prefix} a {destination_dir}")
    os.makedirs(destination_dir, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        if blob.name.endswith('/'):
            continue
        relative_path = os.path.relpath(blob.name, prefix)
        dest_file_path = os.path.join(destination_dir, relative_path)
        os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
        blob.download_to_filename(dest_file_path)
        print(f"✅ {blob.name} → {dest_file_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcs_path', type=str, required=True)   # Ej: 2025_06
    parser.add_argument('--model_path', type=str, required=True) # Ej: 2025_06/outputs
    args = parser.parse_args()

    bucket_name = "proyecto_2_ml_central"
    local_path = "/tmp/eval"

    print("📥 Descargando test set y modelo...")

    download_from_gcs(bucket_name, f"{args.gcs_path}/train", f"{local_path}/train")
    download_from_gcs(bucket_name, f"{args.gcs_path}/val", f"{local_path}/val")
    download_from_gcs(bucket_name, f"{args.gcs_path}/test", f"{local_path}/test")

    download_from_gcs(bucket_name, f"{args.model_path}/modelo_final", f"{local_path}/modelo_final")

    print("📚 Cargando modelo")
    tokenizer = AutoTokenizer.from_pretrained(f"{local_path}/modelo_final")
    model = AutoModelForSequenceClassification.from_pretrained(f"{local_path}/modelo_final")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # ← nuevo

    print('Training args')

    training_args = TrainingArguments(
        output_dir="/tmp/eval",
        disable_tqdm=True,
        logging_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator
    )

    print("📚 Cargando datasets")
    train_dataset = load_from_disk(f"{local_path}/train")
    val_dataset = load_from_disk(f"{local_path}/val")
    test_dataset = load_from_disk(f"{local_path}/test")

    print("🧠 Inferencia sin barra de progreso") # Cambios: 2
    pred_train = trainer.predict(train_dataset)
    pred_val = trainer.predict(val_dataset)
    pred_test = trainer.predict(test_dataset)

    y_true_train = train_dataset["label"]
    y_true_val = val_dataset["label"]
    y_true_test = test_dataset["label"]

    y_pred_train = np.argmax(pred_train.predictions, axis=-1)
    y_pred_val = np.argmax(pred_val.predictions, axis=-1)
    y_pred_test = np.argmax(pred_test.predictions, axis=-1)

    print('Reporte de métricas')

    print("📊 Classification Report: Train")
    print(classification_report(y_true_train, y_pred_train))

    print("📊 Classification Report: Val")
    print(classification_report(y_true_val, y_pred_val))

    print("📊 Classification Report: Test")
    print(classification_report(y_true_test, y_pred_test))

    print('Finalizado')

if __name__ == "__main__":
    main()

