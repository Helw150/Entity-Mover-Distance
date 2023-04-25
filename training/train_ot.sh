export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -n 2 | awk '{ print $NF }')
export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr " " ",")

PORT=$(comm -23 <(seq 1000 2000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

DS=$1

STEPS=3000
E_STEPS=200
if [ $DS == "ZESHEL" ]; then
    STEPS=30000
    E_STEPS=2000
fi

python -m torch.distributed.launch --master_port=$PORT --nproc_per_node=2 train_biencoder.py \
       --model_name_or_path "Intel/ColBERT-NQ" \
       --cache_dir "/data/.cache" \
       --dataset $DS \
       --dist "ot" \
       --output_dir "/data/wheld3/models/ot_biencoder_$DS" \
       --max_steps $STEPS \
       --per_device_train_batch_size 4 \
       --per_device_eval_batch_size 32 \
       --metric_for_best_model="mrr" \
       --load_best_model_at_end=True \
       --gradient_accumulation_steps 2 \
       --evaluation_strategy "steps" \
       --eval_steps $E_STEPS \
       --save_strategy "steps" \
       --save_steps $E_STEPS \
       --weight_decay 0.01 \
       --seed 1 \
       --remove_unused_columns False \
       --data_seed 1 \
       --save_total_limit 1 \
       --learning_rate 0.0001 \
       --evaluation_strategy "steps" \
       --logging_steps 1 \
       --fsdp "full_shard auto_wrap" \
       --fsdp_transformer_layer_cls_to_wrap "BertLayer"
