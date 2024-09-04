
lora_name="lora_flux"
output_dir="flux/outputs"
dataset_config="dataset_config.toml"
script='sd-scripts/tools/cache_latents_flux.py'
processes=1
venv/bin/accelerate launch  --num_processes $processes --mixed_precision bf16 --num_cpu_threads_per_process 1 $script --pretrained_model_name_or_path flux1-dev.safetensors --clip_lclip_l.safetensors --t5xxl t5xxl_fp16.safetensors --ae ae.safetensors --cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers --max_data_loader_n_workers 2 --seed 42 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 --optimizer_type adamw8bit --learning_rate 1e-4 --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base --highvram --max_train_epochs 4 --save_every_n_epochs 1 --dataset_config $dataset_config --output_dir $output_dir --output_name $lora_name --timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 --cache_latents_only
