#!/bin/bash

# Train the models with sdm_camus dataset
# Model configs


source ./scripts/ckpts.sh

train_models=("cris")

datasets=("camus")
camus_prompts=("p0" "p1" "p2" "p3" "p4" "p5" "p6" "p7" )

declare -A prompts_configs
prompts_configs[camus]=${camus_prompts[@]}

# Overwrites of vars
## Pretrained on p6 but fine-tuned on p7
batch_size=32
for model in ${train_models[@]}; do
    for dataset in ${datasets[@]};do
        prompts="${prompts_configs[$dataset]}"
        for prompt in ${prompts[@]}; do
            python src/train.py experiment=${model}.yaml datamodule=img_txt_mask_${dataset}.yaml prompt_type=${prompt} datamodule.batch_size=${batch_size} datamodule.num_workers=8 logger.wandb.project=SDM_CAMUS logger.wandb.name=${model}_${dataset}_pf_ft_${prompt}_reduce_lr tags="[${model}, ${dataset}, pt_ft, fine_tune, reduce_lr ${prompt}]" output_masks_dir=output_masks/${model}/pt_on_sdm_camus_ft_on_camus_reduce_lr/${prompt} ckpt_path=${sdm_camus_cris_reduce_lr_ckpts[$prompt]}
        done
    done
done
