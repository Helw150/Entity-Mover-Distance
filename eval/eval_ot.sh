export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -n 1 | awk '{ print $NF }')
export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr " " ",")

DS=$1

CKPT=$(ls -t /data/wheld3/models/ot_biencoder_$DS/checkpoint-*/pytorch_model.bin | tail -n 1 | awk '{ print $NF }')
echo $CKPT

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python eval_biencoder.py \
       --model_name_or_path  "Intel/ColBERT-NQ" \
       --cache_dir "/data/.cache" \
       --dataset $DS \
       --seed 1 \
       --use_ot_sort true \
       --load_path $CKPT \
      
