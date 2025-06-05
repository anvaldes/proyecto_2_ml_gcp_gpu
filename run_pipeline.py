from google.cloud import aiplatform

# Inicializa el entorno
aiplatform.init(
    project="proyecto-1-461620",
    location="us-central1",
    staging_bucket="gs://proyecto-1-461620-vertex-pipelines-us-central1"
)

# Define el CustomJob
job = aiplatform.CustomJob(
    display_name="xgb-distributed-training",
    worker_pool_specs=[
        {
            "machine_spec": {
                "machine_type": "n1-standard-8",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": "us-central1-docker.pkg.dev/proyecto-1-461620/my-kfp-repo/train-distributed:latest",
                "command": [
                    "accelerate", "launch", "train.py"
                ],
                "args": [
                    "--gcs_path=2025_06",
                    "--output_dir=./results"
                ],
            },
        }
    ],
    base_output_dir="gs://proyecto-1-461620-vertex-pipelines-us-central1/custom-job-outputs",
)

# Ejecuta el job
job.run(service_account="vertex-ai-pipeline-sa@proyecto-1-461620.iam.gserviceaccount.com", sync=True)



