import os
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"

import torch
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

# -------------------------
# 0) 路径设置
# -------------------------
video_save_dir = "/home/ma-user/workspace/yunfeng/DiffSynth-Studio-main/eccv/lora_gaussion"
video_save_dir_f = "/home/ma-user/workspace/yunfeng/DiffSynth-Studio-main/eccv/lora_gaussion/flow_match"
os.makedirs(video_save_dir, exist_ok=True)

# 你的 LoRA（若不想跑第二阶段，后面可以注释掉相关部分）
lora_path = "/cache/yunfeng/eccv/lora2_gaussion_flowmatch/step-1000.safetensors"
# lora_path = "/temp/wyf/second_spatial_local_spatial_pooling_loss_with_latents_target/step-2000.safetensors"

# -------------------------
# 1) 初始化管线（与你的完全一致）
# -------------------------
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    redirect_common_files=False,
    model_configs=[
        ModelConfig(
            model_id="LOCAL",
            origin_file_pattern="/cache/yunfeng/Wan2.2/models_t5_umt5-xxl-enc-bf16.pth",
        ),
        ModelConfig(
            model_id="LOCAL",
            origin_file_pattern="/cache/yunfeng/Wan2.2/diffusion_pytorch_model*.safetensors",
        ),
        ModelConfig(
            model_id="LOCAL",
            origin_file_pattern="/cache/yunfeng/Wan2.2/Wan2.2_VAE.pth",
        ),
    ],
    tokenizer_config=ModelConfig(
        model_id="LOCAL",
        origin_file_pattern="/cache/yunfeng/Wan2.2/google/umt5-xxl/",
    ),
)


# -------------------------
# 2) 10 组 prompts
# -------------------------
items = [
    # {
    #     "name": "elderly_man_closeup_cinematic",
    #     "prompt": "A realistic close-up of an elderly man with silver-gray hair and a soft beard, sitting on sunlit grass. He wears a light linen shirt and a beige vest, his face illuminated by warm afternoon sunlight filtering through the trees. The camera moves from a medium shot to a close-up, capturing defined wrinkles, sun-kissed skin texture, and individual beard strands. His eyes sparkle with vitality, exuding a sense of wisdom and energy. The background is softly blurred with golden bokeh, highlighting the lively and confident expression on his face. High realism, natural lighting, ultra-detailed texture, 8K resolution, cinematic depth of field.",
    #     "neg": "low resolution, blurry, flat colors, unrealistic lighting, toy-like, distorted wings, bad reflections, low detail, smearing, banding, flicker, jitter, text, watermark"
    # },
    # {
    #     "name": "clockwork_robot_rainy_alley",
    #     "prompt": "A single sleek clockwork humanoid robot standing still in a narrow rainy neon alley at night. Brass-and-steel plates with fine engravings, subtle steam vents, and tiny exposed gears under glass panels. Raindrops bead and run along the metal surface, reflecting colorful signage. The robot tilts its head slightly as if listening, minimal motion, cinematic camera push-in. Wet pavement with neon reflections, soft volumetric mist, high realism, ultra-detailed materials, 8K, shallow depth of field, dramatic rim light.",
    #     "neg": "low resolution, blurry, flat colors, unrealistic lighting, plastic/toy-like, bad anatomy, deformed limbs, warped geometry, low detail, compression artifacts, banding, smearing, flicker, jitter, text, watermark, logo, oversharpen, overexposed, underexposed"
    # },
    # {
    #     "name": "gourmet_beef_burger_with_melted_cheese",
    #     "prompt": "A cinematic ultra-detailed food photography shot of a gourmet beef burger with melted cheese, fresh lettuce, tomato slices, glossy brioche bun, golden crispy french fries on the side, steam rising subtly, dramatic studio lighting, shallow depth of field, macro food photography, realistic textures, juicy meat details, soft shadows, dark moody background, professional food styling, 8K resolution, hyper-realistic, masterpiece",
    #     "neg": "low resolution, blurry, flat colors, unrealistic lighting, plastic/toy-like, bad anatomy, deformed hands, low detail, compression artifacts, banding, smearing, flicker, jitter, text, watermark, logo"
    # },
    # {
    #     "name": "A cinematic enchanted forest landscape",
    #     "prompt": "A cinematic enchanted forest landscape, ancient towering trees covered in moss and vines, soft glowing particles floating in the air, sunlight piercing through dense foliage creating dramatic light shafts, a narrow path winding through the forest floor covered with fallen leaves, ultra-detailed bark textures, realistic leaf translucency, atmospheric fog, shallow depth of field, cinematic color grading, volumetric lighting, global illumination, photorealistic fantasy style, 8K resolution, ultra high detail, masterpiece quality",
    #     "neg": "low resolution, blurry, flat colors, unrealistic lighting, plastic/toy-like, bad reflections, low detail, compression artifacts, banding, smearing, flicker, jitter, text, watermark, logo"
    # },
    # {
    #     "name": "white_owl_snowy_pine_fog",
    #     "prompt": "A single white owl perched on a frost-covered pine branch in a snowy forest at dawn. Feathers rendered with crisp micro-detail, visible barbs and soft down, gentle breath mist in the cold air. Light snowfall drifting past the lens, pale blue fog weaving between dark pine trunks. The owl blinks slowly and turns its head with calm precision. Cinematic close-up with creamy bokeh, natural cold lighting, ultra-detailed texture, 8K, shallow depth of field.",
    #     "neg": "low resolution, blurry, flat colors, unrealistic lighting, cartoonish, deformed beak, extra wings, bad feathers, low detail, compression artifacts, banding, smearing, flicker, jitter, text, watermark, logo"
    # },

]
items = [
    {
        "name": "golden_koi_pond_autumn_maple",
        "prompt": "Several elegant golden koi fish swimming beneath crystal-clear water in a tranquil Japanese garden pond during autumn. Rippling reflections of fiery red maple leaves shimmer across the surface, while soft sunlight filters through drifting morning mist. Every koi scale rendered with ultra-fine metallic detail, subtle water distortion and realistic caustic lighting dancing across smooth stones below. Fallen maple leaves float gently on the water, tiny ripples expanding outward naturally. Cinematic composition with shallow depth of field, atmospheric fog, photorealistic textures, soft bokeh, ultra-detailed environment, natural lighting, 8K, masterpiece quality.",
        "neg": "low resolution, blurry, flat colors, unrealistic lighting, cartoonish, deformed beak, extra wings, bad feathers, low detail, compression artifacts, banding, smearing, flicker, jitter, text, watermark, logo"
    },
    {
        "name": "crystal_waterfall_mountain_valley",
        "prompt": "A majestic crystal-clear waterfall cascading down steep cliffs in a hidden mountain valley at sunrise. Fine mist rises into the golden morning light, creating glowing rainbow particles in the air. Moss-covered rocks glisten with moisture, while shallow streams weave through lush green grass and wildflowers. Hyper-detailed water simulation with realistic reflections and flowing motion, cinematic wide-angle composition, atmospheric depth, volumetric lighting, photorealistic textures, ultra-detailed environment, shallow depth of field, 8K, masterpiece quality.",
        "neg": "low resolution, blurry, flat colors, unrealistic lighting, cartoonish, deformed beak, extra wings, bad feathers, low detail, compression artifacts, banding, smearing, flicker, jitter, text, watermark, logo"
    },
    {
        "name": "floating_island_fantasy_sky",
        "prompt": "A breathtaking floating island suspended high above the clouds during golden hour, covered with lush forests, cascading waterfalls, and ancient stone ruins wrapped in ivy. Warm sunlight illuminates drifting clouds beneath the island while birds glide through the glowing sky. Tiny waterfalls dissolve into mist before reaching the clouds below. Cinematic fantasy atmosphere, ultra-detailed foliage, realistic rock textures, volumetric sunlight, atmospheric perspective, photorealistic rendering, soft cloud scattering, 8K, shallow depth of field, masterpiece quality.",
        "neg": "low resolution, blurry, flat colors, unrealistic lighting, cartoonish, deformed beak, extra wings, bad feathers, low detail, compression artifacts, banding, smearing, flicker, jitter, text, watermark, logo"
    },
    {
        "name": "cozy_cabin_snowstorm_night",
        "prompt": "A warm wooden cabin glowing softly in the middle of a snowy forest during a winter night storm. Thick snow falls heavily under dim lantern light while warm orange light shines through frosted windows. Snow accumulates naturally on rooftops and pine branches, with visible wind patterns swirling through the air. Cinematic winter atmosphere, ultra-detailed snow textures, realistic lighting contrast, atmospheric fog, photorealistic environment, shallow depth of field, 8K resolution, masterpiece quality.",
        "neg": "low resolution, blurry, flat colors, unrealistic lighting, cartoonish, deformed beak, extra wings, bad feathers, low detail, compression artifacts, banding, smearing, flicker, jitter, text, watermark, logo"
    },
    {
        "name": "ancient_desert_temple_sunset",
        "prompt": "An ancient sandstone temple emerging from endless desert dunes at sunset, illuminated by dramatic golden-orange light. Fine sand drifts across worn stone carvings and weathered pillars, while distant heat haze distorts the horizon subtly. Intricate hieroglyphic details carved into the temple walls, cinematic scale, atmospheric dust particles floating in the air, realistic sand textures, volumetric sunset lighting, photorealistic rendering, ultra-detailed environment, 8K, masterpiece quality.",
        "neg": "low resolution, blurry, flat colors, unrealistic lighting, cartoonish, deformed beak, extra wings, bad feathers, low detail, compression artifacts, banding, smearing, flicker, jitter, text, watermark, logo"
    },
    {
        "name": "luxury_steak_fire_grill_closeup",
        "prompt": "An ultra-detailed cinematic close-up of a premium grilled steak resting on a dark stone plate beside glowing fire embers. Juices glisten across the perfectly seared surface, with visible seasoning crystals and delicate smoke rising naturally. Melted butter slowly drips across the steak while rosemary and garlic roast nearby. Dramatic warm lighting, shallow depth of field, macro food photography, realistic textures, photorealistic reflections, dark moody background, 8K resolution, masterpiece quality.",
        "neg": "low resolution, blurry, flat colors, unrealistic lighting, cartoonish, deformed beak, extra wings, bad feathers, low detail, compression artifacts, banding, smearing, flicker, jitter, text, watermark, logo"
    },
    {
        "name": "strawberry_pancake_honey_drizzle",
        "prompt": "A stack of fluffy strawberry pancakes on a rustic wooden table during soft morning sunlight. Golden honey slowly drizzles over fresh strawberries and whipped cream, creating realistic glossy reflections and smooth flowing motion. Powdered sugar dust floats gently through warm sunlight, while tiny butter melts naturally across the pancake surface. Cinematic breakfast photography, ultra-detailed textures, shallow depth of field, soft bokeh, photorealistic food styling, 8K resolution, masterpiece quality.",
        "neg": "low resolution, blurry, flat colors, unrealistic lighting, cartoonish, deformed beak, extra wings, bad feathers, low detail, compression artifacts, banding, smearing, flicker, jitter, text, watermark, logo"
    },
    {
        "name": "dark_chocolate_cake_rainy_window",
        "prompt": "A luxurious dark chocolate cake placed beside a rainy café window during an overcast afternoon. Rich chocolate glaze reflects soft ambient light while droplets slide down the window glass in the blurred background. Fresh berries and delicate mint leaves decorate the top with hyper-realistic texture detail. Atmospheric moody lighting, cinematic close-up composition, shallow depth of field, realistic reflections, ultra-detailed dessert photography, photorealistic rendering, 8K, masterpiece quality.",
        "neg": "low resolution, blurry, flat colors, unrealistic lighting, cartoonish, deformed beak, extra wings, bad feathers, low detail, compression artifacts, banding, smearing, flicker, jitter, text, watermark, logo"
    },
    {
        "name": "lantern_festival_river_night",
        "prompt": "Hundreds of glowing lanterns floating gently across a calm river during a traditional night festival. Warm golden light reflects beautifully on rippling water while soft fog drifts between distant wooden bridges and lantern-lit buildings. Tiny sparks rise into the cool night air, creating cinematic atmosphere and depth. Photorealistic reflections, ultra-detailed lighting effects, atmospheric perspective, shallow depth of field, volumetric fog, cinematic composition, 8K resolution, masterpiece quality.",
        "neg": "low resolution, blurry, flat colors, unrealistic lighting, cartoonish, deformed beak, extra wings, bad feathers, low detail, compression artifacts, banding, smearing, flicker, jitter, text, watermark, logo"
    },
    {
        "name": "cyberpunk_rain_city_alley",
        "prompt": "A neon-lit cyberpunk alleyway during heavy rain at midnight, filled with glowing holographic signs and reflective wet pavement. Vibrant blue and magenta neon lights bounce realistically across puddles while steam rises from underground vents. A futuristic motorcycle parked beside illuminated storefronts adds cinematic storytelling. Hyper-detailed urban textures, atmospheric fog, realistic rain simulation, shallow depth of field, photorealistic rendering, dramatic cinematic lighting, 8K resolution, masterpiece quality.",
        "neg": "low resolution, blurry, flat colors, unrealistic lighting, cartoonish, deformed beak, extra wings, bad feathers, low detail, compression artifacts, banding, smearing, flicker, jitter, text, watermark, logo"
    }
]

# -------------------------
# 3) 批量参数（你可以统一改这里）
# -------------------------
seed0 = 999
fps = 15
quality = 4

t2v_h, t2v_w = 704, 1280
num_frames = 121
tiled = True
denoising_strength = 0.7

# -------------------------
# 4) 先生成一波 T2V（704×1280）
# -------------------------
# for i, it in enumerate(items, start=1):
#     name = it["name"]
#     prompt = it["prompt"]
#     neg = it["neg"]
#     seed = seed0 + i  # 每个视频用不同 seed，方便多样性；想固定就改成 seed0

#     out_low = os.path.join(video_save_dir, f"{name}_704p.mp4")
#     print(f"[T2V] {name} -> {out_low}")

#     video_low = pipe(
#         prompt=prompt,
#         negative_prompt=neg,
#         seed=seed,
#         tiled=tiled,
#         height=t2v_h,
#         width=t2v_w,
#         num_frames=num_frames,
#         base=True
#     )
#     save_video(video_low, out_low, fps=fps, quality=quality)


# -------------------------
# 5) 再生成一波 V2V + LoRA（1408×2560）
# -------------------------
# sr_h,  sr_w  = 2176,3840
# sr_h,  sr_w  = 1472,2560
sr_h,  sr_w  = 1728,3072

pipe.load_lora(pipe.dit, lora_path)

for i, it in enumerate(items, start=1):
    name = it["name"]
    prompt = it["prompt"]
    neg = it["neg"]
    seed = seed0 + i

    in_low = os.path.join(video_save_dir, f"{name}_704p.mp4")
    out_sr = os.path.join(video_save_dir_f, f"{name}_1080P.mp4")

    print(f"[V2V+LoRA] {name} <- {in_low} -> {out_sr}")

    # input_video_data = VideoData(in_low, height=sr_h, width=sr_w)

    video_sr = pipe(
        prompt=prompt,
        negative_prompt=neg,
        # input_video=input_video_data,
        denoising_strength=denoising_strength,
        seed=seed,
        tiled=tiled,
        height=sr_h,
        width=sr_w,
        num_frames=num_frames,
        base=False
    )
    save_video(video_sr, out_sr, fps=fps, quality=quality)

print("Done.")
