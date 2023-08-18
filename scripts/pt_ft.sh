#!/bin/bash

#####################################################################################
# Finetune the VLSMs with CAMUS dataset, already finetuned on SDM CAMUS dataset:    #
#####################################################################################

# Experiment configs
train_models=("clip_seg" "cris")
pretrained_dataset="sdm_camus"
dataset="camus"
prompts=("p0" "p1" "p2" "p3" "p4" "p5" "p6" "p7" )


for model in ${train_models[@]}; do
    if [ $model == "clip_seg" ]
    then
        batch_size=128
        lr=0.002
    else
        batch_size=32
        lr=0.00002
    fi
    for prompt in ${prompts[@]}; do

        if [ ${prompt} == "p7" ] 
        then
            # The sdm camus data lacks p7 prompt so we use p6
            sdm_ckpt=logs/train/runs/${model}_${pretrained_dataset}_p6/checkpoints/best.ckpt
        else
            sdm_ckpt=logs/train/runs/${model}_${pretrained_dataset}_${prompt}/checkpoints/best.ckpt
        fi
        python src/train.py \
            experiment=${model}.yaml \
            experiment_name=${model}_${pretrained_dataset}_pt_${dataset}_ft_${prompt} \
            datamodule=img_txt_mask_${dataset}.yaml \
            prompt_type=${prompt} \
            datamodule.batch_size=${batch_size} \
            tags="[${model}, ${dataset}, ${prompt}]" \
            output_masks_dir=output_masks/${model}/${pretrained_dataset}_pt_${dataset}_ft/${prompt} \
            ckpt_path=${sdm_ckpt} \
            trainer.accelerator=gpu \
            trainer.devices=[0] \
            trainer.precision=16-mixed
    done
done
