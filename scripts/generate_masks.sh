train_models=("clip_seg")

sdm_camus_prompts=("p0" "p1" "p2" "p3" "p4" "p5" "p6" )
dataset="camus"

declare -A clip_seg_ckpts
clip_seg_ckpts[p0]="logs/train/runs/2023-06-29_17-58-46/checkpoints/epoch_035.ckpt"
clip_seg_ckpts[p1]="logs/train/runs/2023-06-26_12-48-15/checkpoints/epoch_020.ckpt"
clip_seg_ckpts[p2]="logs/train/runs/2023-06-26_19-02-55/checkpoints/epoch_025.ckpt"
clip_seg_ckpts[p3]="logs/train/runs/2023-06-27_02-04-05/checkpoints/epoch_026.ckpt"
clip_seg_ckpts[p4]="logs/train/runs/2023-06-29_15-16-02/checkpoints/epoch_029.ckpt"
clip_seg_ckpts[p5]="logs/train/runs/2023-06-29_16-53-33/checkpoints/epoch_014.ckpt"
clip_seg_ckpts[p6]="logs/train/runs/2023-06-29_22-22-35/checkpoints/epoch_017.ckpt"


for model in "${train_models[@]}"; do
    if [ $model == "cris" ]; then
        batch_size=32 # Because of the larger memory footprint in cris
    else
        batch_size=128 # Because of the smaller memory footprint in clip_seg
    fi
    for prompt in ${sdm_camus_prompts[@]}; do
        ckpt_path=${clip_seg_ckpts[$prompt]}
        python src/eval.py experiment=${model}.yaml datamodule=img_txt_mask_${dataset}.yaml prompt_type=${prompt} datamodule.batch_size=${batch_size} logger=csv.yaml output_masks_dir=output_masks/${model}/full_train/${dataset}/${prompt} ckpt_path=${ckpt_path}
    done
done


train_models=("cris")

sdm_camus_prompts=("p0" "p1" "p2" "p3" "p4" "p5" "p6")
dataset="sdm_camus"

declare -A cris_ckpts
cris_ckpts[p0]="logs/train/runs/2023-06-27_02-32-18/checkpoints/epoch_009.ckpt"
cris_ckpts[p1]="logs/train/runs/2023-06-26_12-47-38/checkpoints/epoch_009.ckpt"
cris_ckpts[p2]="logs/train/runs/2023-06-26_15-27-59/checkpoints/epoch_011.ckpt"
cris_ckpts[p3]="logs/train/runs/2023-06-26_18-20-00/checkpoints/epoch_011.ckpt"
cris_ckpts[p4]="logs/train/runs/2023-06-26_21-11-05/checkpoints/epoch_009.ckpt"
cris_ckpts[p5]="logs/train/runs/2023-06-26_23-51-26/checkpoints/epoch_009.ckpt"
cris_ckpts[p6]="logs/train/runs/2023-06-29_19-46-20/checkpoints/epoch_009.ckpt"

for model in "${train_models[@]}"; do
    if [ $model == "cris" ]; then
        batch_size=32 # Because of the larger memory footprint in cris
    else
        batch_size=128 # Because of the smaller memory footprint in clip_seg
    fi
    for prompt in ${sdm_camus_prompts[@]}; do
        ckpt_path=${cris_ckpts[$prompt]}
        python src/eval.py experiment=${model}.yaml datamodule=img_txt_mask_${dataset}.yaml prompt_type=${prompt} datamodule.batch_size=${batch_size} logger=csv.yaml output_masks_dir=output_masks/${model}/full_train/${dataset}/${prompt} ckpt_path=${ckpt_path}
    done
done