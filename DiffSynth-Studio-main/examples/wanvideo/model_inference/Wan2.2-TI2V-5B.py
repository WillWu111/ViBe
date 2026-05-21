import os
from random import seed
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"   # 禁止下载（可选但推荐）
# os.environ["HF_HOME"] = "/cache/yunfeng/hf"      # 顺手避免 transformers 用默认缓存（可选）

import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

# 创建视频保存目录
video_save_dir = "/home/ma-user/workspace/yunfeng/DiffSynth-Studio-main/videos"
os.makedirs(video_save_dir, exist_ok=True)


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    # 关键：关掉公共文件重定向，否则 .pth 可能被改去找 .safetensors
    redirect_common_files=False,
    model_configs=[
        ModelConfig(
            model_id="LOCAL",  # 写啥都行，反正我们用绝对路径
            origin_file_pattern="/cache/yunfeng/Wan2.2/models_t5_umt5-xxl-enc-bf16.pth",
            # **vram_config
        ),
        ModelConfig(
            model_id="LOCAL",
            origin_file_pattern="/cache/yunfeng/Wan2.2/diffusion_pytorch_model*.safetensors",
            # **vram_config
        ),
        ModelConfig(
            model_id="LOCAL",
            origin_file_pattern="/cache/yunfeng/Wan2.2/Wan2.2_VAE.pth",
            # **vram_config
        ),
    ],

    tokenizer_config=ModelConfig(
        model_id="LOCAL",
        origin_file_pattern="/cache/yunfeng/Wan2.2/google/umt5-xxl/",
    ),
)



# video = pipe(
#     prompt = "A realistic close-up of an elderly man with gray hair and a thick gray beard, wearing a light-colored shirt. His head is slightly lowered. The camera zooms from full body to close-up, highlighting detailed facial wrinkles, skin texture, forehead lines, eye bags, and beard strands. High resolution, cinematic lighting, sharp details.",
#     negative_prompt = "repeating patterns, Blurry face, low detail, distorted features, extra limbs, cartoon style, smooth plastic skin, low resolution, flat colors, lack of texture",
#     seed=0, tiled=True,
#     height=704, width=1280,
#     num_frames=121,
# )
# # save_video(video, os.path.join(video_save_dir, "1080P_temporal_local_all_pooling_3000iter_spatial_local_spatial_pooling(training2_inference4)_lora2_step_3000iter.mp4"), fps=15, quality=5)
# save_video(video, os.path.join(video_save_dir, "old_man.mp4"), fps=15, quality=5)
# path = os.path.join(video_save_dir, "old_man.mp4")
lora_path = "/temp/wyf/noinward/step-500.safetensors" 
lora_scale = 1.2  # LoRA强度
pipe.load_lora(pipe.dit, lora_path)
input_video_data = VideoData("/temp/wyf/old_man.mp4", height=1088, width=1920)
video = pipe(
    prompt = "A realistic close-up of an elderly man with gray hair and a thick gray beard, wearing a light-colored shirt. His head is slightly lowered. The camera zooms from full body to close-up, highlighting detailed facial wrinkles, skin texture, forehead lines, eye bags, and beard strands. High resolution, cinematic lighting, sharp details.",
    negative_prompt = "repeating patterns, Blurry face, low detail, distorted features, extra limbs, cartoon style, smooth plastic skin, low resolution, flat colors, lack of texture",
    input_video=input_video_data,  # 传入加载的视频数据
    denoising_strength=0.7,  # 控制生成视频与输入视频的相似度，0-1之间，值越小越接近原视频
    seed=0, tiled=True,
    height=1088, width=1920,
    num_frames=121,
    base=False
)
save_video(video, os.path.join(video_save_dir, "old_man_1080P_v2v_relay_lora_noinward_step_500.mp4"), fps=15, quality=5)

