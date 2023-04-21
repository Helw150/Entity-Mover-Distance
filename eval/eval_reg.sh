export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -n 1 | awk '{ print $NF }')
export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr " " ",")

CKPT=$(ls -t /data/wheld3/models/regular_biencoder_ECB/checkpoint-*/pytorch_model.bin | tail -n 1 | awk '{ print $NF }')
echo $CKPT

DS=$1

python eval_biencoder.py \
       --model_name_or_path "Intel/ColBERT-NQ" \
       --cache_dir "/data/.cache" \
       --dataset $DS \
       --load_path $CKPT \
      
