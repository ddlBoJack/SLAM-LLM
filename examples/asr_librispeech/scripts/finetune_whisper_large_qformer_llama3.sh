#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=/home/yxdu/hit/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=/home/yxdu/hit/speech/models/whisper/large-v3.pt
encoder_path_hf=/home/yxdu/hit/speech/models/whisper-large-v3
llm_path=/home/yxdu/hit/speech/models/Meta-Llama-3-8B-Instruct
train_data_path=/home/yxdu/hit/speech/data/common/4/en/train.jsonl
val_data_path=/home/yxdu/hit/speech/data/common/4/en/test.jsonl

output_dir=/home/yxdu/hit/speech/output/asr-largev3-qformer-lama3-521

# 使用find命令搜索所有.pt文件，并获取最后修改日期最晚的文件
latest_file=$(find "$output_dir" -type f -name "*.pt" -printf '%T+ %p\n' | sort -r | head -n 1 | tail -n 1 | cut -d" " -f2-)

# 检查是否找到了文件
if [[ -n "$latest_file" ]]; then
    ckpt_dir=$(dirname "$latest_file")
    peft_ckpt="$ckpt_dir"
    ckpt_name="$latest_file"
    # ckpt_name=/home/yxdu/hit/speech/output/st-lama3-asr/asr/6/model_202405130533_1539.pt
    echo $ckpt_name
else
    echo "No .pt files found in $output_dir."
fi






hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=Meta-Llama-3-8B-Instruct \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=whisper \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_path_hf=$encoder_path_hf \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=q-former \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=mel \
++dataset_config.mel_size=128 \
++dataset_config.fix_length_audio=80 \
++train_config.model_name=asr \
++train_config.num_epochs=3 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=1000000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=6 \
++train_config.val_batch_size=6 \
++train_config.num_workers_dataloader=8 \
++train_config.output_dir=$output_dir \
++metric=acc \
"






# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 5 \
        --master_port=29503 \
        $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=false \
        ++fsdp_config.pure_bf16=true \
        $hydra_args
fi
        # ++model_config.ckpt_path=$ckpt_name \