from kfp import compiler
from pipeline_distributed import distributed_pipeline

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=distributed_pipeline,
        package_path="distributed_pipeline.json"
    )
    print("✅ Pipeline compilado como 'distributed_pipeline.json'")
