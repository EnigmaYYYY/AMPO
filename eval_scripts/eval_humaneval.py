from evalscope.evalscope.run import run_task
from evalscope.evalscope.config import TaskConfig

def main(model_name):
    model = model_name
    task_cfg = TaskConfig(
        model=model,
        eval_type="openai_api",
        api_url="http://0.0.0.0:8000/v1", # we use vllm api to deploy our models
        api_key="EMPTY",
        datasets=['humaneval'],
        eval_batch_size=4,
        generation_config={
            'max_tokens': 8192,
            'temperature': 0.6,
            'top_p': 1.0,
            'n': 1,
            'seed': 42,
        },
        repeats=1,
        use_sandbox=True,
        sandbox_type='docker',
        sandbox_manager_config={
            'base_url': 'your remote sandbox manager URL'
        },
        judge_worker_num=8, 
    )

    run_task(task_cfg=task_cfg)
    
if __name__ == "__main__":
    import fire
    fire.Fire(main)