#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
# export PATH=$PATH:/usr/local/cuda-11.8/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
# export CUDA_HOME=/usr/local/cuda-11.8


# export PYTHONPATH=/root/fairseq:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=7291c67639a70b6aff97fede6add8b8516c7e079
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=/home/yxdu/hit/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=/home/yxdu/hit/speech/models/whisper/large-v2.pt
encoder_path_hf=/home/yxdu/hit/speech/models/whisper-large-v2
llm_path=/home/yxdu/hit/speech/models/Qwen1.5-7B
train_data_path=/home/yxdu/hit/speech/data/common/4/en/test.jsonl
val_data_path=/home/yxdu/hit/speech/data/common/4/en/test.jsonl


checkpoint_dir=/home/yxdu/hit/speech/output/whisper-qformer-qwen1.5-7b-cn-all-527-bleu
output_dir=/home/yxdu/hit/speech/bleu_output
# 使用find命令搜索所有.pt文件，并获取最后修改日期最晚的文件
latest_file=$(find "$checkpoint_dir" -type f -name "*.pt" -printf '%T+ %p\n' | sort -r | head -n 1 | tail -n 1 | cut -d" " -f2-)

# 检查是否找到了文件
if [[ -n "$latest_file" ]]; then
    ckpt_dir=$(dirname "$latest_file")
    peft_ckpt="$ckpt_dir"
    ckpt_name="$latest_file"
    # ckpt_name=/home/yxdu/hit/speech/output/st-lama3-asr/asr/6/model_202405130533_1539.pt
    echo $ckpt_name
else
    echo "No .pt files found in $checkpoint_dir."
fi
ckpt_dir=$output_dir
decode_log=$ckpt_dir/decode__beam4

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_dir \
        ++model_config.llm_name="vicuna-7b-v1.5" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=4096 \
        ++model_config.encoder_name=whisper \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_path_hf=$encoder_path_hf \
        ++model_config.encoder_dim=1280 \
        ++model_config.encoder_projector=q-former \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=mel \
        ++dataset_config.fix_length_audio=80 \
        ++dataset_config.mel_size=80 \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=16 \
        ++train_config.num_workers_dataloader=16 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++model_config.ckpt_path=$ckpt_name \
        # ++peft_ckpt=$ckpt_path \
        # ++train_config.use_peft=true \
        # ++train_config.peft_config.r=32 \
        # ++dataset_config.normalize=true \
        # ++model_config.encoder_projector=q-former \
        # ++dataset_config.fix_length_audio=64 \
