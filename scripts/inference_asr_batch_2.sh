#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/SLAM-LLM/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

cd /SLAM-LLM

speech_encoder_path=/cxgroup/model/whisper/large-v3.pt
llm_path=/cxgroup/model/TowerInstruct-7B-v0.2

output_dir=/exps/tower-7b-finetune-asr-linear-lora-none-steplrwarmupkeep1e-4-whisper-largev3-it-LID-longprompt-average-20240321-test/asr/average
ckpt_path=$output_dir
peft_ckpt=$ckpt_path
val_data_path=/data/italian/test.jsonl
decode_log=$ckpt_path/decode_log_test_beam4

# -m debugpy --listen 5678 --wait-for-client
python src/slam_llm/pipeline/inference_batch.py \
--config-path "/SLAM-LLM/scripts/conf" \
--config-name "asr_vicuna_lora.yaml" \
hydra.run.dir=$ckpt_path \
++model_config.llm_name="tower-7b" \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=whisper \
++dataset_config.normalize=true \
++model_config.normalize=true \
++model_config.encoder_name=whisper \
++model_config.encoder_ds_rate=2 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=linear \
++dataset_config.dataset=speech_dataset \
++dataset_config.prompt="Transcribe speech to italian text. Output the transcription directly without redundant content. Ensure that the output is not duplicated." \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=mel \
++dataset_config.mel_size=128 \
++dataset_config.inference_mode=true \
++train_config.model_name=asr \
++train_config.batching_strategy=custom \
++train_config.num_epochs=1 \
++train_config.val_batch_size=4 \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++ckpt_path=$ckpt_path/model.pt \
++decode_log=$decode_log \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
#++peft_ckpt=$peft_ckpt \
# ++model_config.encoder_projector=q-former \
# ++dataset_config.fix_length_audio=64 \
# --peft_ckpt $peft_ckpt \
# --use_peft --peft_method lora \