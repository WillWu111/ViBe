# **ViBe: Ultra-High-Resolution Video Synthesis Born from Pure Images** 🎥✨

[![arXiv](https://img.shields.io/badge/arXiv-2603.23326-b31b1b.svg)](https://arxiv.org/pdf/2603.23326)

<!-- Teaser 放在比较靠前的位置 -->
![Teaser – Ultra-High-Resolution Results](assets/teaser.jpg)

<p align="center"><em>Teaser: ViBe generates ultra-high-resolution videos with rich details and coherent global structure. ✨</em></p>

## **Introduction** 🎬

**ViBe** presents an **image-only training framework** for **ultra-high-resolution video generation**, aiming to unlock **4K video synthesis** from pre-trained video Diffusion Transformers without expensive high-resolution video training. Instead of relying on large-scale high-resolution video data, ViBe explores how a low-resolution video generator can be effectively adapted using only image supervision.

A central challenge in this setting is the **modality gap between images and videos**. Directly fine-tuning a video model on high-resolution images often leads to artifacts and residual noise during video inference. To address this, ViBe introduces **Relay LoRA**, a two-stage adaptation strategy that separates **modality alignment** from **spatial extrapolation**, enabling high-resolution generation while reducing image-induced degradation.

To further improve generation quality, ViBe proposes **Global-Coarse-Local-Fine (GCLF) Attention**, which combines local fine-grained modeling with compact global context aggregation. This design improves detail fidelity while maintaining semantic consistency across the entire frame.

ViBe also introduces a **High-Frequency-Awareness Training Objective (HFATO)** to enhance the recovery of fine textures and sharp structures from degraded latent inputs. This leads to clearer and more detailed high-resolution video synthesis.

Extensive experiments on modern video DiTs, such as **Wan2.2**, show that ViBe can generate **4K videos** with strong semantic coherence and rich visual details, achieving **state-of-the-art performance on VBench**.

## **Methodology** 🧠

<!-- 方法架构图放在 Methodology 下面 -->
![ViBe Framework](assets/method.jpg)

<p align="center"><em>Figure: Overview of the ViBe framework. 🧩</em></p>

## **Installation** 📦

ViBe requires `torch==2.7.1` (for flex attention), and other packages listed in `requirements.txt`. You can set up a new experiment with:

```bash
conda create -n ViBe python=3.12
conda activate ViBe
cd DiffSynth-Studio-main
pip install -e .
```

Clone this repo to your project directory:

```bash
git clone https://github.com/WillWu111/ViBe.git
```

## **Models** 🤖

We provide our trained LoRA models for ultra-high-resolution video synthesis. You can download them from Hugging Face:

| Model | Description | Download |
|-------|-------------|----------|
| ViBe LoRA | Relay LoRA for high-resolution video generation | [Download](https://huggingface.co/yunfengWu/ViBe) |

## **Inference** 🚀

You can specify `target_height`, `target_width`, and `spatial_down_factor` as command-line arguments:

```bash
python ViBe.py --target_height 1408 --target_width 2560 --spatial_down_factor 2
```

**Note:** Ensure that the latent height and width are divisible by `spatial_down_factor`.

## **Training** 🏋️

Our training loss functions are defined in [diffsynth/diffusion/loss.py](diffsynth/diffusion/loss.py). ViBe uses a two-stage training approach with different loss functions:

### Stage 1: Modality Alignment (First Stage)

```python
# First Stage: Standard noise prediction
def FlowMatchSFTLoss(pipe, **inputs):
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id]

    noise = torch.randn_like(inputs["input_latents"])
    x_noisy = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    loss = mse_loss(noise_pred, training_target)
    return loss
```

### Stage 2: Spatial Extrapolation (Second Stage)

```python
# Second Stage: x0 prediction with down-up cycle (HFATO)
def FlowMatchSFTLoss(pipe, **inputs):
    # Down-sample and up-sample x0 to create degraded input
    x0_down = F.interpolate(x0, scale_factor=(1.0, 0.5, 0.5), mode="trilinear")
    x0_up = F.interpolate(x0_down, size=(T, H, W), mode="trilinear")

    # Add noise to degraded input
    x_noisy = pipe.scheduler.add_noise(x0_up, noise, timestep)

    # Predict flow and recover x0
    flow_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    x0_pred = x_noisy - sigma * flow_pred

    # Loss: predict original x0 (not degraded x0_up)
    loss = mse_loss(x0_pred, x0)
    return loss
```

**Key Difference:** Stage 1 trains on original latents, while Stage 2 trains the model to recover high-frequency details from degraded latent inputs, enhancing the model's ability to synthesize fine textures.

## **Results** 🏆

**ViBe** outperforms previous state-of-the-art methods on **VBench**, with significant improvements in aesthetic appeal, imaging quality, and overall consistency.  🌈



## **Citation** 📚

If you use **ViBe** in your research, please cite our paper:

```bibtex
@misc{wu2026vibeultrahighresolutionvideosynthesis,
      title={ViBe: Ultra-High-Resolution Video Synthesis Born from Pure Images}, 
      author={Yunfeng Wu and Hongying Cheng and Zihao He and Songhua Liu},
      year={2026},
      eprint={2603.23326},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.23326}, 
}