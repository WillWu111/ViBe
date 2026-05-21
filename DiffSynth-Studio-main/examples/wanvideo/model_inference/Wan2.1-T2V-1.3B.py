import os
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


# -------------------------
# 0) 路径设置
# -------------------------
video_save_dir = "/home/ma-user/workspace/yunfeng/DiffSynth-Studio-main/eccv/wan2.1_1.3b/704P"
os.makedirs(video_save_dir, exist_ok=True)
video_save_dir_4k = "/home/ma-user/workspace/yunfeng/DiffSynth-Studio-main/eccv/wan2.1_1.3b/4k"
os.makedirs(video_save_dir_4k, exist_ok=True)

# 你的 LoRA（若不想跑第二阶段，后面可以注释掉相关部分）
lora_path = "/cache/yunfeng/eccv_wan2.1/lora2/step-2000.safetensors"

# -------------------------
# 1) 初始化管线（与你的完全一致）
# -------------------------
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="LOCAL", origin_file_pattern="/cache/hongying/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="LOCAL", origin_file_pattern="/cache/hongying/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="LOCAL", origin_file_pattern="/cache/hongying/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="LOCAL", origin_file_pattern="/cache/hongying/wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/"),
)


items = [
    {
        "name": "enhanced_armored_wolf_forest",
        "prompt": "A majestic wolf standing proudly in a forest clearing, wearing intricately designed armor that gleams in the sunlight. The wolf's detailed armor is made of dark, polished metal with gold accents, reflecting its fierce and noble nature. The focus is on the wolf's expressive face and finely crafted armor, showing intricate patterns and textures, with special attention to the rich detailing of its fur and the gleam of sunlight on the metal. The camera zooms in for a closer shot, emphasizing the richness of the wolf’s features, with the armor's reflections and the individual strands of fur clearly visible. The dense forest surrounding the wolf has towering trees with soft beams of sunlight filtering through the canopy, casting dynamic shadows on the ground. The atmosphere is heroic and powerful, ultra-detailed textures, 8K.",
        "neg": "low resolution, blurry, flat colors, unrealistic lighting, toy-like, distorted wings, bad reflections, low detail, smearing, banding, flicker, jitter, text, watermark"
    },
    # {
    #     "name": "cyberpunk_neon_ninja_city",
    #     "prompt": "A stealthy cyberpunk ninja standing on a rain-soaked rooftop overlooking a neon-lit futuristic city at night. The ninja wears a sleek, high-tech suit with glowing blue circuitry patterns and a reflective visor. Raindrops fall continuously, creating ripples and reflections across the metallic surfaces. The camera slowly pans closer, highlighting the intricate suit details, water droplets sliding off the armor, and neon lights reflecting across the wet environment. Massive holographic billboards flicker in the background, with flying vehicles passing through the skyline. The atmosphere is moody, cinematic, ultra-detailed textures, dynamic lighting, volumetric fog, 8K resolution.",
    #     "neg": "low resolution, blurry, flat lighting, overexposed highlights, unrealistic reflections, stiff pose, bad anatomy, flickering lights, noise, text, watermark"
    # },
    {
        "name": "majestic_white_tiger_snow",
        "prompt": "A powerful white tiger walking slowly through a snowy forest during a gentle snowfall. Its thick fur is highly detailed, with visible individual strands covered in snowflakes. The tiger’s breath condenses into mist in the cold air. The camera tracks alongside the tiger at a low angle, emphasizing its strength and graceful movement. Snow crunches softly under its paws, leaving clear footprints behind. Tall pine trees surround the scene, their branches heavy with snow. Soft diffused light creates a calm yet majestic atmosphere. Ultra-realistic textures, cinematic slow motion, shallow depth of field, 8K.",
        "neg": "low resolution, blurry, cartoonish, flat lighting, bad anatomy, flicker, noise, artifacts, text, watermark"
    },
    {
        "name": "dolphins_sunset_ocean_jump",
        "prompt": "A group of dolphins leaping out of the ocean during a vibrant sunset. Water droplets scatter in the air, catching golden sunlight and forming sparkling arcs. The camera follows the dolphins in a smooth slow-motion shot, capturing the elegance of their movement. The ocean surface reflects the orange and pink sky, with gentle waves rolling beneath. Seabirds fly in the distance, and the horizon glows warmly. The atmosphere is lively, peaceful, and full of motion. Ultra-detailed water simulation, cinematic lighting, 8K.",
        "neg": "low detail, unrealistic water, stiff motion, blur, noise, bad reflections, artifacts, text, watermark"
    },
    {
        "name": "cheese_pizza_pull_melt",
        "prompt": "A freshly baked cheese pizza being pulled apart, with melted cheese stretching dramatically between slices. The crust is golden and crispy, with slight charring on the edges. The camera captures the moment in slow motion, focusing on the stretchy cheese strands and bubbling surface. Steam rises from the hot pizza, and oil glistens under warm lighting. The background is softly blurred, emphasizing the food. Ultra-detailed textures, macro shot, cinematic lighting, 8K.",
        "neg": "low resolution, dull colors, unrealistic cheese, blur, noise, artifacts, text, watermark"
    },
    {
        "name": "ocean_cliff_waves_crash",
        "prompt": "Powerful ocean waves crashing against a rugged coastal cliff under dramatic cloudy skies. Water splashes high into the air, forming detailed droplets and mist. The camera is positioned near the cliff edge, slightly shaking to simulate natural movement. The waves roll in rhythm, with foam patterns forming and dissolving. Dark clouds move quickly across the sky, with occasional sunlight breaking through. The atmosphere is intense and cinematic. Ultra-realistic water physics, dynamic lighting, 8K.",
        "neg": "low resolution, unrealistic water, flat shading, blur, noise, artifacts, text, watermark"
    }
]


seed0 = 0
t2v_h, t2v_w = 704, 1280
sr_h,  sr_w  = 2176, 3840
num_frames = 121

# Text-to-video
# for i, it in enumerate(items, start=1):
#     name = it["name"]
#     prompt = it["prompt"]
#     neg = it["neg"]
#     seed = seed0 + i  # 每个视频用不同 seed，方便多样性；想固定就改成 seed0

#     out_low = os.path.join(video_save_dir, f"{name}_704p.mp4")
#     print(f"[T2V] {name} -> {out_low}")

#     video = pipe(
#         prompt=prompt,
#         negative_prompt=neg,
#         seed=seed, tiled=True,
#         height=t2v_h, width=t2v_w, num_frames=num_frames,
#         base=True
#     )
#     print(f"base video saved")
#     save_video(video, out_low, fps=15, quality=5)
#     print("保存成功！")

# Video-to-video
pipe.load_lora(pipe.dit, lora_path)

for i, it in enumerate(items, start=1):
    name = it["name"]
    prompt = it["prompt"]
    neg = it["neg"]
    seed = seed0 + i

    in_low = os.path.join(video_save_dir, f"{name}_704p.mp4")
    out_sr = os.path.join(video_save_dir_4k, f"{name}_4k.mp4")

    print(f"[V2V+LoRA] {name} <- {in_low} -> {out_sr}")

    input_video_data = VideoData(in_low, height=sr_h, width=sr_w)
    video = pipe(
        prompt=prompt,
        negative_prompt=neg,
        input_video=input_video_data, 
        denoising_strength=0.7,
        seed=seed, tiled=True,
        height=sr_h, width=sr_w, num_frames=num_frames,
        base=False
    )
    print(f"lora video saved")
    save_video(video, out_sr, fps=15, quality=5)
    print("保存成功！")