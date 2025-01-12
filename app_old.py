import torch
import sys
import os
import subprocess
import shutil
import tempfile
import uuid
import gradio as gr
from glob import glob
from huggingface_hub import snapshot_download
from tqdm.notebook import tqdm  # Для красивого вывода в Colab

print("🚀 Инициализация SVFR...")

# Download models
os.makedirs("models", exist_ok=True)
print("\n📥 Загрузка основных моделей...")

snapshot_download(
    repo_id = "fffiloni/SVFR",
    local_dir = "./models",
    tqdm_class=tqdm  # Используем tqdm для прогресс-бара
)

# List of subdirectories to create inside "checkpoints"
subfolders = [
    "stable-video-diffusion-img2vid-xt"
]
# Create each subdirectory
for subfolder in subfolders:
    os.makedirs(os.path.join("models", subfolder), exist_ok=True)

print("\n📥 Загрузка модели Stable Video Diffusion...")
snapshot_download(
    repo_id = "stabilityai/stable-video-diffusion-img2vid-xt",
    local_dir = "./models/stable-video-diffusion-img2vid-xt",
    tqdm_class=tqdm  # Используем tqdm для прогресс-бара
)

print("\n✅ Все модели загружены успешно!")

def infer(lq_sequence, task_name):
    try:
        # Очищаем память GPU перед запуском
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        unique_id = str(uuid.uuid4())
        output_dir = f"results_{unique_id}"

        if task_name == "BFR":
            task_id = "0"
        elif task_name == "colorization":
            task_id = "1"
        elif task_name == "BFR + colorization":
            task_id = "0,1"
        
        # Устанавливаем переменные окружения для процесса
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
        env['CUDA_LAUNCH_BLOCKING'] = '1'
        env['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
        
        # Запускаем inference с настройками памяти
        subprocess.run(
            [
                "python", "infer.py",
                "--config", "config/infer.yaml",
                "--task_ids", f"{task_id}",
                "--input_path", f"{lq_sequence}",
                "--output_dir", f"{output_dir}",
            ],
            check=True,
            env=env
        )

        # Search for the mp4 file in a subfolder of output_dir
        output_video = glob(os.path.join(output_dir,"*.mp4"))
        print(output_video)
        
        if output_video:
            output_video_path = output_video[0]  # Get the first match
        else:
            output_video_path = None
        
        print(output_video_path)
        return output_video_path
    
    except subprocess.CalledProcessError as e:
        # Очищаем память при ошибке
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        raise gr.Error(f"Error during inference: {str(e)}")
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        raise gr.Error(f"Unexpected error: {str(e)}")
    finally:
        # Очищаем старые результаты
        try:
            for old_dir in glob("results_*"):
                if os.path.isdir(old_dir):
                    shutil.rmtree(old_dir, ignore_errors=True)
        except Exception as e:
            print(f"Error cleaning up: {e}")

css="""
div#col-container{
    margin: 0 auto;
    max-width: 982px;
}
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# SVFR: A Unified Framework for Generalized Video Face Restoration")
        gr.Markdown("SVFR is a unified framework for face video restoration that supports tasks such as BFR, Colorization, Inpainting, and their combinations within one cohesive system.")
        gr.HTML("""
        <div style="display:flex;column-gap:4px;">
            <a href="https://github.com/wangzhiyaoo/SVFR">
                <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
            </a> 
            <a href="https://wangzhiyaoo.github.io/SVFR/">
                <img src='https://img.shields.io/badge/Project-Page-green'>
            </a>
            <a href="https://arxiv.org/pdf/2501.01235">
                <img src='https://img.shields.io/badge/ArXiv-Paper-red'>
            </a>
            <a href="https://huggingface.co/spaces/fffiloni/SVFR-demo?duplicate=true">
                <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-sm.svg" alt="Duplicate this Space">
            </a>
            <a href="https://huggingface.co/fffiloni">
                <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-me-on-HF-sm-dark.svg" alt="Follow me on HF">
            </a>
        </div>
        """)
        with gr.Row():
            with gr.Column():
                input_seq = gr.Video(label="Video LQ")
                task_name = gr.Radio(
                    label="Task", 
                    choices=["BFR", "colorization", "BFR + colorization"], 
                    value="BFR"
                )
                submit_btn = gr.Button("Submit")
            with gr.Column():
                output_res = gr.Video(label="Restored")
                gr.Examples(
                    examples = [
                        ["./assert/lq/lq1.mp4", "BFR"],
                        ["./assert/lq/lq2.mp4", "BFR + colorization"],
                        ["./assert/lq/lq3.mp4", "colorization"]
                    ],
                    inputs = [input_seq, task_name]
                )
    
    submit_btn.click(
        fn = infer,
        inputs = [input_seq, task_name],
        outputs = [output_res]
    )

demo.queue().launch(show_api=False, show_error=True)
