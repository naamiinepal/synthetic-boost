#!/bin/bash

# Train the models with sdm_camus dataset
# Model configs


train_models=("clip_seg")

camus_prompts=("p0" "p1" "p2" "p3" "p4" "p5" "p6" "p7" )
sdm_camus_prompts=("p0" "p1" "p2" "p3" "p4" "p5" "p6" )
datasets=("camus" "sdm_camus")

declare -A prompts_configs
prompts_configs[camus]=${camus_prompts[@]}
prompts_configs[sdm_camus]=${sdm_camus_prompts[@]}

# Overwrites of vars
## Pretrained on p6 but fine-tuned on p7
batch_size=32
for model in ${train_models[@]}; do
    for dataset in ${datasets[@]};do 
        prompts="${prompts_configs[$dataset]}"
        for prompt in ${prompts[@]}; do
            python src/train.py experiment=${model}.yaml datamodule=img_txt_mask_${dataset}.yaml prompt_type=${prompt} datamodule.batch_size=${batch_size} datamodule.num_workers=8 logger.wandb.project=SDM_CAMUS logger.wandb.name=${model}_${dataset}_full_train_${prompt}_reduce_lr tags="[${model}, ${dataset}, fine_tune, reduce_lr ${prompt}]" output_masks_dir=output_masks/${model}/full_train_reduce_lr/${dataset}/${prompt} trainer.devices=[1]
        done
    done
done
