#!/bin/bash

emb_dim=16; mlp_dims="1024 512 256"; optimizer="adam"; batch_size=1e4; # do not change
dataset="criteo_2"; model="dnn"; emb_type="qr"; lr=1e-3; l2=3e-6; 
epoch=8; val_per_epoch=1; early_stop=3; gpu=0; gpu_id=0;

qr_ratio=2; # final parameter

gpus=(0 1 2 3)
ratios=(2 3 4 5 6)
models=("dnn" "dcn" "deepfm" "ipnn")

for b in 0; do
  # qr_ratio=${ratios[$b]};
for a in 0; do
  model=${models[$a]};

  gpu=${gpus[$gpu_id]};
  gpu_id=$((gpu_id + 1))
  log_path="./log/${dataset}/${model}/${emb_type}/"; log_name="${emb_type}+ratio=${qr_ratio}";
  
  python -u main.py \
    --qr_ratio ${qr_ratio} \
    --dataset ${dataset} --model ${model} --emb_type ${emb_type} --lr ${lr} --l2 ${l2} \
    --epoch ${epoch} --val_per_epoch ${val_per_epoch} --early_stop ${early_stop} --gpu ${gpu} \
    --emb_dim ${emb_dim} --mlp_dims ${mlp_dims} --optimizer ${optimizer} --batch_size ${batch_size} \
    --log_path ${log_path} --log_name ${log_name} & 
done
wait
gpu_id=0;
done
