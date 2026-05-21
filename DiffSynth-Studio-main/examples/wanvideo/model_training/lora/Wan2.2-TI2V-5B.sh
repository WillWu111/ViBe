export DIFFSYNTH_SKIP_DOWNLOAD=true
accelerate launch /home/ma-user/workspace/yunfeng/DiffSynth-Studio-main/examples/wanvideo/model_training/train.py \
  --dataset_base_path /temp/processed_data/InternImgs \
  --dataset_metadata_path /temp/processed_data/InternImgs/hr_data_final_clean.csv \
  --height 1536 \
  --width 2752 \
  --num_frames 1 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "LOCAL:/cache/yunfeng/eccv/lora1_gaussion/final.pth,LOCAL:/cache/yunfeng/Wan2.2/models_t5_umt5-xxl-enc-bf16.pth,LOCAL:/cache/yunfeng/Wan2.2/Wan2.2_VAE.pth" \
  --tokenizer_path "/cache/yunfeng/Wan2.2/google/umt5-xxl/" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/cache/yunfeng/eccv/lora2_gaussion_flowmatch" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --save_steps 1000\
  --extra_inputs "input_image"