#!/bin/bash

ROOT='Your ROOT PATH'

cd $ROOT/AMPO/eval_scripts


# 定义模型路径、名称、模板的数组
MODEL_NAME=(
    "Qwen2.5-7B-Ins-AMPO"
)

# 遍历所有模型
for i in "${!MODEL_NAME[@]}"; do
    MODEL_NAME=${MODEL_NAME[$i]}

    echo "Running inference for $MODEL_NAME ..."

    python evalscope.py \
      --model_path "$MODEL_NAME" \
    
    sleep 60
done