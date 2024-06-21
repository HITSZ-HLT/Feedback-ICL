python run_llm_hf_pre.py \
    --run_pre \
    --task ATSC \
    --dataset rest \
    --model llama-2-13b-chat \
    --train_idx 1 \
    --mode train \
    --write_to_train \
    --device cuda:3 \
    --batch_size 16