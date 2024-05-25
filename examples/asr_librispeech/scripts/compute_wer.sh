# #cd /root/SLAM-LLM

# trans="/home/yxdu/hit/sjtu/SLAM-LLM/common/output/st-zh-2/asr/2/asr_gt"
# preds="/home/yxdu/hit/sjtu/SLAM-LLM/common/output/st-zh-2/asr/2/asr_pred"


# python src/llama_recipes/utils/whisper_tn.py ${trans} ${trans}.proc
# python src/llama_recipes/utils/llm_tn.py ${preds} ${preds}.proc
# python src/llama_recipes/utils/compute_wer.py ${trans}.proc ${preds}.proc ${preds}.proc.wer

# tail -3 ${preds}.proc.wer

# # echo "-------num2word------"
# # python src/llama_recipes/utils/num2word.py ${preds}.proc ${preds}.proc.words
# # python src/llama_recipes/utils/compute_wer.py ${trans} ${preds}.proc.words ${preds}.proc.wer.words

# # tail -3 ${preds}.proc.wer.words






#!/bin/bash

# 定义文件夹路径
dir="/home/yxdu/hit/speech/asr_wer/en"
# 定义输出结果文件
results_file="$dir/summary_results.txt"

# 遍历目录中所有后缀为 _gt 的文件
for gt_file in "$dir"/*_gt; do
    # 检查对应的 _pred 文件是否存在
    pred_file="${gt_file%_gt}_pred"
    
    if [ -f "$pred_file" ]; then
        # 如果对应的 _pred 文件存在

        echo "Processing $gt_file and $pred_file..."

        # 执行处理步骤
        # 处理 _gt 文件
        python /home/yxdu/hit/SLAM-LLM/src/slam_llm/utils/whisper_tn.py "$gt_file" "${gt_file}.proc"
        # 处理 _pred 文件
        python /home/yxdu/hit/SLAM-LLM/src/slam_llm/utils/llm_tn.py "$pred_file" "${pred_file}.proc"
        # 计算处理后文件的 WER
        python /home/yxdu/hit/SLAM-LLM/src/slam_llm/utils/compute_wer.py "${gt_file}.proc" "${pred_file}.proc" "${pred_file}.proc.wer"

        # 将结果和文件名写入结果文件
        echo "Results for $pred_file:" >> $results_file
        tail -3 "${pred_file}.proc.wer" >> $results_file
        echo "Processed files:" >> $results_file
        echo "GT File: $gt_file" >> $results_file
        echo "PRED File: $pred_file" >> $results_file
        echo "------------------------" >> $results_file

    else
        echo "No matching prediction file for $gt_file,file is missing $pred_file"
    fi
done































# 遍历目录中所有后缀为 _gt 的文件
# for gt_file in "$dir"/*_gt_en; do
#     # 检查对应的 _pred 文件是否存在
#     pred_file="${gt_file%_gt_en}_pred_en"
    
#     if [ -f "$pred_file" ]; then
#         # 如果对应的 _pred 文件存在

#         echo "Processing $gt_file and $pred_file..."

#         # 执行处理步骤
#         # 处理 _gt 文件
#         python src/llama_recipes/utils/whisper_tn.py "$gt_file" "${gt_file}.proc"
#         # 处理 _pred 文件
#         python src/llama_recipes/utils/llm_tn.py "$pred_file" "${pred_file}.proc"
#         # 计算处理后文件的 WER
#         python src/llama_recipes/utils/compute_wer.py "${gt_file}.proc" "${pred_file}.proc" "${pred_file}.proc.wer"

#         # 将结果和文件名写入结果文件
#         echo "Results for $pred_file:" >> $results_file
#         tail -3 "${pred_file}.proc.wer" >> $results_file
#         echo "Processed files:" >> $results_file
#         echo "GT File: $gt_file" >> $results_file
#         echo "PRED File: $pred_file" >> $results_file
#         echo "------------------------" >> $results_file

#     else
#         echo "No matching prediction file for $gt_file,file is missing $pred_file"
#     fi
# done
