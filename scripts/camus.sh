#!/bin/bash

# Train the models with sdm_camus dataset
# Model configs


train_models=("clip_seg" "cris")

dataset="camus"
prompts=("p0" "p1" "p2" "p3" "p4" "p5" "p6" "p7")

for model in ${train_models[@]}; do
    if [ $train_models == "clip_seg" ]
    then
        batch_size=128
        lr=0.002
    else
        batch_size=32
        lr=0.00002
    fi

    for prompt in ${prompts[@]}; do
        
        python src/train.py \
            experiment=${model}.yaml \
            experiment_name=${model}_${dataset}_${prompt} \
            datamodule=img_txt_mask_${dataset}.yaml \
            prompt_type=${prompt} \
            datamodule.batch_size=${batch_size} \
            model.optimizer.lr=${lr} \
            tags="[${model}, ${dataset}, ${prompt}]" \
            output_masks_dir=output_masks/${model}/${dataset}/${prompt} \
            trainer.accelerator=gpu \
            trainer.devices=[0] \
            trainer.precision=16-mixed\
            trainer.max_epochs=2
    done
done
