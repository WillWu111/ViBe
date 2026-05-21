import argparse
import os

import torch
from safetensors.torch import save_file
from diffsynth.core import load_state_dict

from diffsynth.diffusion.training_module import DiffusionTrainingModule
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"



def main():
   
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

    if pipe.dit is None:
        raise ValueError("No DIT model was loaded. Please check --model_paths or --model_id_with_origin_paths.")

    _pre_l1 = sum(p.detach().float().abs().sum().item() for p in pipe.dit.parameters())
    pipe.load_lora(pipe.dit, "/cache/yunfeng/eccv/lora1_gaussion/step-1000.safetensors", alpha=1.0, hotload=False)
    print("LoRA fuse 后相对 fuse 前是否有参数变化:", _pre_l1 != sum(p.detach().float().abs().sum().item() for p in pipe.dit.parameters()))
    state_dict = {name: weight.detach().cpu() for name, weight in pipe.dit.state_dict().items()}

    print(f"Saving merged DIT weights")
    os.makedirs(os.path.dirname(os.path.abspath("/cache/yunfeng/eccv/lora1_gaussion/final.pth")), exist_ok=True)
    torch.save(state_dict, "/cache/yunfeng/eccv/lora1_gaussion/final.pth")
    print(f"Merged DIT weights saved")
    ################dolora merge#####################
    # merge_dora_to_base_model(
    #     pipe.dit,
    #     "/cache/yunfeng/eccv/lora1_gaussion/step-1000.safetensors",
    #     alpha=1.0,
    # )
    # torch.save({name: weight.detach().cpu() for name, weight in pipe.dit.state_dict().items()}, "/cache/yunfeng/eccv/lora1_guassion/final.pth")


def merge_dora_to_base_model(model, lora_path, alpha=1.0):
    lora = load_state_dict(lora_path, torch_dtype=torch.float32, device="cuda")
    modules = dict(model.named_modules())

    updated = 0
    skipped = 0
    max_change = 0.0

    for b_key, weight_up in lora.items():
        if ".lora_B." not in b_key:
            continue

        name = b_key.split(".lora_B.")[0]
        a_key = b_key.replace(".lora_B.", ".lora_A.")
        m_key = b_key.replace(".lora_B.", ".lora_magnitude_vector.")

        if name not in modules or a_key not in lora or m_key not in lora:
            skipped += 1
            continue

        module = modules[name]
        if not hasattr(module, "weight"):
            skipped += 1
            continue

        base_weight = module.weight.data.float()
        weight_down = lora[a_key].float()
        magnitude = lora[m_key].float()

        delta = torch.mm(weight_up.float(), weight_down)

        if base_weight.shape != delta.shape:
            print(f"[skip shape mismatch] {name}: base={tuple(base_weight.shape)}, lora={tuple(delta.shape)}")
            skipped += 1
            continue

        before = base_weight.clone()
        merged = base_weight + alpha * delta
        norm = torch.linalg.norm(merged, dim=1).clamp(min=1e-6)
        dora_factor = (magnitude / norm).view(-1, 1)
        final_weight = dora_factor * merged

        module.weight.data.copy_(final_weight.to(module.weight.dtype))
        max_change = max(max_change, (final_weight - before).abs().max().item())
        updated += 1

    print(f"[DoRA merge] updated={updated}, skipped={skipped}, max_change={max_change:.6e}")
    print("[DoRA merge] 是否有权重变化:", max_change > 0)



if __name__ == "__main__":
    main()
