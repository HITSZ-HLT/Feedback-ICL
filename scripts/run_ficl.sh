# Feedback-ICL (ours)
python run_llm_hf_test.py \
    --run_ficl \
    --task ATSC \
    --dataset rest \
    --model llama-2-13b-chat \
    --icl_mode bm25 \
    --train_idx 1 \
    --device cuda:3 \
    --batch_size 16

# Conventional ICL
python run_llm_hf_test.py \
    --run_icl \
    --task ATSC \
    --dataset rest \
    --model llama-2-13b-chat \
    --icl_mode bm25 \
    --train_idx 1 \
    --device cuda:3 \
    --batch_size 16