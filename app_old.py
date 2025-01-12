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

# Download models
os.makedirs("models", exist_ok=True)

snapshot_download(
    repo_id = "fffiloni/SVFR",
    local_dir = "./models"  
)

# List of subdirectories to create inside "checkpoints"
subfolders = [
    "stable-video-diffusion-img2vid-xt"
]
# Create each subdirectory
for subfolder in subfolders:
    os.makedirs(os.path.join("models", subfolder), exist_ok=True)

# snapshot_download(
#     repo_id = "stabilityai/stable-video-diffusion-img2vid-xt",
#     local_dir = "./models/stable-video-diffusion-img2vid-xt"  
# )

def infer(lq_sequence, task_name):
    
    unique_id = str(uuid.uuid4())
    output_dir = f"results_{unique_id}"

    if task_name == "BFR":
        task_id = "0"
    elif task_name == "colorization":
        task_id = "1"
    elif task_name == "BFR + colorization":
        task_id = "0,1"
    
    try:
        # Run the inference command
        subprocess.run(
            [
                "python", "infer.py",
                "--config", "config/infer.yaml",
                "--task_ids", f"{task_id}",
                "--input_path", f"{lq_sequence}",
                "--output_dir", f"{output_dir}",
            ],
            check=True
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
        raise gr.Error(f"Error during inference: {str(e)}")

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