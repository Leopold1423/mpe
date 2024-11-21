#!/bin/bash

emb_dim=16; mlp_dims="1024 512 256"; optimizer="adam"; batch_size=1e4; # do not change
dataset="kdd12_2"; model="dnn"; emb_type="optfs"; lr=1e-3; l2=0.0; 
epoch=1; val_per_epoch=2; early_stop=1; gpu=0; gpu_id=0;

optfs_l1=1e-9; mask_init=0.1; tau=1e-2; # final parameter

gpus=(0 1 2 3)
l1s=(1e-10 2e-10)
models=("dnn" "dcn" "deepfm" "ipnn")

for c in 0; do
  # optfs_l1=${l1s[$c]};
for a in 0; do
  model=${models[$a]};

  gpu=${gpus[$gpu_id]};
  gpu_id=$((gpu_id + 1))
  log_path="./log/${dataset}/${model}/${emb_type}/"; log_name="${emb_type}+mask_init=${mask_init}+tau=${tau}+optfs_l1=${optfs_l1}";

  python -u main.py \
    --mask_init ${mask_init} --optfs_l1 ${optfs_l1} --tau ${tau} \
    --dataset ${dataset} --model ${model} --emb_type ${emb_type} --lr ${lr} --l2 ${l2} \
    --epoch ${epoch} --val_per_epoch ${val_per_epoch} --early_stop ${early_stop} --gpu ${gpu} \
    --emb_dim ${emb_dim} --mlp_dims ${mlp_dims} --optimizer ${optimizer} --batch_size ${batch_size} \
    --log_path ${log_path} --log_name ${log_name} & 
done
wait
gpu_id=0;
done