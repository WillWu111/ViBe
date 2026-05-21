import os
import argparse
from random import seed
import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from huggingface_hub import hf_hub_download

# Parse command line arguments
parser = argparse.ArgumentParser(description="ViBe: Video Generation Pipeline")
parser.add_argument("--target_height", type=int, default=1408, help="Target video height (default: 1408)")
parser.add_argument("--target_width", type=int, default=2560, help="Target video width (default: 2560)")
parser.add_argument("--spatial_down_factor", type=int, default=2, help="Spatial downsample factor (default: 2)")
args = parser.parse_args()

os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
video_save_dir = "./videos"
os.makedirs(video_save_dir, exist_ok=True)


LORA_REPO_ID = "yunfengWu/ViBe"
LORA_FILENAME = "ViBe.safetensors"
lora_cache_dir = "./lora_cache"
os.makedirs(lora_cache_dir, exist_ok=True)
lora_path = hf_hub_download(repo_id=LORA_REPO_ID, filename=LORA_FILENAME, cache_dir=lora_cache_dir)



pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

num_frames = 121
base_height = 704
base_width = 1280

target_height = args.target_height
target_width = args.target_width

video = pipe(
    prompt="A realistic close-up of an elderly man with silver-gray hair and a soft beard, sitting on sunlit grass. He wears a light linen shirt and a beige vest, his face illuminated by warm afternoon sunlight filtering through the trees. The camera moves from a medium shot to a close-up, capturing defined wrinkles, sun-kissed skin texture, and individual beard strands. His eyes sparkle with vitality, exuding a sense of wisdom and energy. The background is softly blurred with golden bokeh, highlighting the lively and confident expression on his face. High realism, natural lighting, ultra-detailed texture, 8K resolution, cinematic depth of field.",
    negative_prompt="low resolution, blurry, flat colors, unrealistic lighting, toy-like, distorted wings, bad reflections, low detail, smearing, banding, flicker, jitter, text, watermark",
    height=base_height, width=base_width,
    num_frames=1,
    base=True,
)
save_video(video, os.path.join(video_save_dir,"base.mp4"), fps=15, quality=5)


pipe.load_lora(pipe.dit, lora_path)


input_video = VideoData(os.path.join(video_save_dir,"base.mp4"), height=target_height, width=target_width)
video = pipe(
    input_video=input_video,
    prompt="A realistic close-up of an elderly man with silver-gray hair and a soft beard, sitting on sunlit grass. He wears a light linen shirt and a beige vest, his face illuminated by warm afternoon sunlight filtering through the trees. The camera moves from a medium shot to a close-up, capturing defined wrinkles, sun-kissed skin texture, and individual beard strands. His eyes sparkle with vitality, exuding a sense of wisdom and energy. The background is softly blurred with golden bokeh, highlighting the lively and confident expression on his face. High realism, natural lighting, ultra-detailed texture, 8K resolution, cinematic depth of field.",
    negative_prompt="low resolution, blurry, flat colors, unrealistic lighting, toy-like, distorted wings, bad reflections, low detail, smearing, banding, flicker, jitter, text, watermark",
    height=target_height, width=target_width,
    num_frames=1,
    base=False,
    spatial_down_factor=args.spatial_down_factor,
)
save_video(video, os.path.join(video_save_dir,"ViBe.mp4"), fps=15, quality=5)
