export DIFFSYNTH_SKIP_DOWNLOAD=true
accelerate launch /home/ma-user/workspace/yunfeng/DiffSynth-Studio-main/examples/wanvideo/model_training/train.py \
  --dataset_base_path /temp/processed_data/InternImgs \
  --dataset_metadata_path /temp/processed_data/InternImgs/hr_data_final_clean.csv \
  --height 720 \
  --width 1280 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "LOCAL:/cache/hongying/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model*.safetensors,LOCAL:/cache/hongying/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth,LOCAL:/cache/hongying/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --tokenizer_path "/cache/hongying/wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/cache/yunfeng/eccv_wan2.1/dolora1" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32\
  --save_steps 1000
