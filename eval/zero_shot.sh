export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -n 1 | awk '{ print $NF }')
export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr " " ",")

DS=$1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for MODEL in "Intel/ColBERT-NQ" #"bert-base-uncased" "roberta-base" "google/electra-base-discriminator" "microsoft/deberta-v3-base" 
do
    echo $MODEL
    python eval_biencoder.py \
	   --model_name_or_path $MODEL \
	   --cache_dir "/data/.cache" \
	   --dataset $DS \
	   --seed 1 \
	   --neighbors 64 \
	   --use_ot_sort true
done

