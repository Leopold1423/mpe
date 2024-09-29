#!/bin/bash

emb_dim=16; mlp_dims="1024 512 256"; optimizer="adam"; batch_size=1e4; # do not change
dataset="kdd12_2"; model="dnn"; emb_type="optfp"; lr=1e-3; l2=0.0;
epoch=1; val_per_epoch=2; early_stop=1; gpu=0; gpu_id=0;

bitsets="0123456"; tau=3e-3; group=128; l2_gamma=3e-6; # final parameter

gpus=(0 1 2 3)
l2s=(1e-6 3e-6 6e-6 1e-5)
models=("dnn" "dcn" "deepfm" "ipnn")

for b in 0; do
  # l2_gamma=${l2s[$a]};
for a in 0; do
  model=${models[$a]};

  gpu=${gpus[$gpu_id]};
  gpu_id=$((gpu_id + 1))
  log_path="./log/${dataset}/${model}/${emb_type}/"; log_name="${emb_type}+l2=${l2_gamma}";
  
  python -u main.py \
    --bitsets ${bitsets} --l2_gamma ${l2_gamma} --tau ${tau} --group ${group} \
    --dataset ${dataset} --model ${model} --emb_type ${emb_type} --lr ${lr} --l2 ${l2} \
    --epoch ${epoch} --val_per_epoch ${val_per_epoch} --early_stop ${early_stop} --gpu ${gpu} \
    --emb_dim ${emb_dim} --mlp_dims ${mlp_dims} --optimizer ${optimizer} --batch_size ${batch_size} \
    --log_path ${log_path} --log_name ${log_name} & 
done
wait
gpu_id=0;
done
