import os
import argparse
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from google.cloud import storage
import time
import json
import warnings
import torch

warnings.filterwarnings("ignore")

def validate_hf_dataset(path):
    print(f"🔎 Validando dataset en: {path}")
    
    # Verifica que existan los 3 archivos requeridos
    required_files = ["dataset_info.json", "state.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(path, file)):
            raise FileNotFoundError(f"❌ Falta {file} en {path}")

    # Verifica JSON válido
    with open(os.path.join(path, "dataset_info.json")) as f:
        content = f.read()
        if not content.strip():
            raise ValueError("❌ dataset_info.json está vacío")
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ dataset_info.json inválido: {e}")

    # Verifica que exista al menos un archivo .arrow
    arrow_files = [f for f in os.listdir(path) if f.endswith(".arrow")]
    if not arrow_files:
        raise FileNotFoundError("❌ No se encontró ningún archivo .arrow")

    print("✅ Validación completada")

def download_from_gcs(bucket_name, prefix, destination_dir):
    print(f"🔽 Descargando de gs://{bucket_name}/{prefix} a {destination_dir}")
    os.makedirs(destination_dir, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        if blob.name.endswith('/'):
            continue
        # Mantener estructura relativa exacta respecto al prefijo
        relative_path = os.path.relpath(blob.name, prefix)
        dest_file_path = os.path.join(destination_dir, relative_path)
        os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
        blob.download_to_filename(dest_file_path)
        print(f"✅ {blob.name} → {dest_file_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcs_path', type=str, required=True)  # Ej: proyecto_2_ml_central/2025_06
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    gcs_prefix = args.gcs_path
    output_dir = args.output_dir
    local_path = "/tmp/data"

    print("📥 Descargando datasets desde GCS con API oficial...")
    bucket_name = "proyecto_2_ml_central"

    # ✅ NUEVO: Mensaje sobre uso de GPU o CPU
    print(f"🧠 GPU disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🚀 Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚙️ Usando CPU")

    download_from_gcs(bucket_name, f"{gcs_prefix}/train", f"{local_path}/train")
    download_from_gcs(bucket_name, f"{gcs_prefix}/val", f"{local_path}/val")

    print("📚 Cargando datasets...")
    validate_hf_dataset(f"{local_path}/train")
    validate_hf_dataset(f"{local_path}/val")

    tokenized_train = load_from_disk(f"{local_path}/train")
    tokenized_val = load_from_disk(f"{local_path}/val")

    print("🔧 Cargando modelo y tokenizer desde ./modelo_base")
    tokenizer = AutoTokenizer.from_pretrained("./modelo_base")
    model = AutoModelForSequenceClassification.from_pretrained("./modelo_base", num_labels=2)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch", # Cambiado
        save_steps=0,
        logging_steps=25,
        report_to="none",
        logging_strategy = "no", # Cambiado
        disable_tqdm = True # Cambiado
    )

    print("🚀 Entrenando modelo...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        # tokenizer=tokenizer, Eliminado
        data_collator=data_collator,
    )

    start = time.time()

    trainer.train()

    end = time.time()
    delta = end - start

    hours, rem = divmod(delta, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"Tiempo entrenamiento: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    print("💾 Guardando modelo...")
    trainer.save_model(f"{output_dir}/modelo_final")
    tokenizer.save_pretrained(f"{output_dir}/modelo_final")

    print("📤 Subiendo resultados a GCS...")
    # Subir resultados usando el cliente GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(output_dir):
        for file in files:
            local_path_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_path_file, output_dir)
            blob = bucket.blob(f"{gcs_prefix}/outputs/{relative_path}")
            blob.upload_from_filename(local_path_file)
            print(f"✅ Subido: {blob.name}")

if __name__ == "__main__":
    main()

