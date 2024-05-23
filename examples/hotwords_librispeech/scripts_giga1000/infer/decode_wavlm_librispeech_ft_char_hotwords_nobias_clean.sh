#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

run_dir=/root/SLAM-LLM
cd $run_dir
code_dir=examples/hotwords_librispeech

speech_encoder_path=/nfs/yangguanrou.ygr/ckpts/wavlm_large_ft_giga1000/wavlm_large_ft_giga1000_char.pt
llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5

output_dir=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-giga1000-ft-char-hotwords-20240521
ckpt_path=$output_dir/asr_epoch_1_step_8000
split=librispeech_test_clean
val_data_path=/nfs/maziyang.mzy/data/librispeech/${split}.jsonl
decode_log=$ckpt_path/decode_${split}_beam4_nobias

# -m debugpy --listen 5678 --wait-for-client
python -m debugpy --listen 5678 --wait-for-client $code_dir/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name="vicuna-7b-v1.5" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=4096 \
        ++model_config.encoder_name=wavlm \
        ++model_config.normalize=true \
        ++dataset_config.normalize=true \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=1024 \
        ++model_config.encoder_projector=cov1d-linear \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=raw \
        ++dataset_config.inference_mode=true \
        ++dataset_config.infer_type=nobias \
        ++dataset_config.dataset=gigahotwords_dataset \
        ++dataset_config.file=src/slam_llm/datasets/giga_hotwordsinfer_dataset.py:get_speech_dataset \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=4 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \
        # ++peft_ckpt=$ckpt_path \
        # ++train_config.use_peft=true \
        # ++train_config.peft_config.r=32 \
        # ++dataset_config.normalize=true \
        # ++model_config.encoder_projector=q-former \
        # ++dataset_config.fix_length_audio=64 \