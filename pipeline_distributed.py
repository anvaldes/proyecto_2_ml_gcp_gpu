from kfp import dsl

@dsl.component(
    base_image='us-central1-docker.pkg.dev/proyecto-1-461620/my-kfp-repo/train-distributed:latest',
    packages_to_install=["google-cloud-storage", "transformers", "datasets", "scikit-learn", "torch", "accelerate", "pandas"]
)
def training_step(gcs_path: str, output_dir: str):
    import subprocess
    subprocess.run([
        'accelerate', 'launch', 'train.py',
        f'--gcs_path={gcs_path}',
        f'--output_dir={output_dir}'
    ], check=True)

@dsl.component(
    base_image='us-central1-docker.pkg.dev/proyecto-1-461620/my-kfp-repo/train-distributed:latest',
    packages_to_install=["google-cloud-storage", "transformers", "datasets", "scikit-learn", "torch", "pandas"]
)
def evaluation_step(gcs_path: str, model_path: str):
    import subprocess
    subprocess.run([
        'python3', 'evaluate.py',
        f'--gcs_path={gcs_path}',
        f'--model_path={model_path}'
    ], check=True)

@dsl.pipeline(name="xgb-distributed-training-and-eval")
def distributed_pipeline(
    gcs_path: str = "2025_06",
    output_dir: str = "./results"
):
    train = training_step(gcs_path=gcs_path, output_dir=output_dir)
    evaluate = evaluation_step(gcs_path=gcs_path, model_path=f"{gcs_path}/outputs")
    evaluate.after(train)

