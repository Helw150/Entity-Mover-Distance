DS=$1

python -m torch.distributed.launch --nproc_per_node=4 train_biencoder.py \
       --model_name_or_path "vespa-engine/colbert-medium" \
       --cache_dir "/data/.cache" \
       --dataset $DS \
       --output_dir "/data/wheld3/models/regular_biencoder_$DS" \
       --num_train_epochs 50 \
       --per_device_train_batch_size 8 \
       --per_device_eval_batch_size 16 \
       --metric_for_best_model="f1" \
       --load_best_model_at_end=True \
       --gradient_accumulation_steps 1 \
       --evaluation_strategy "steps" \
       --eval_steps 200 \
       --save_strategy "steps" \
       --save_steps 200 \
       --weight_decay 0.001 \
       --seed 1 \
       --remove_unused_columns False \
       --data_seed 1 \
       --save_total_limit 1 \
       --learning_rate 0.0001 \
       --warmup_ratio 0.03 \
       --lr_scheduler_type "cosine" \
       --evaluation_strategy "steps" \
       --logging_steps 1 \
       --fsdp "full_shard auto_wrap" \
       --fsdp_transformer_layer_cls_to_wrap "BertLayer"
