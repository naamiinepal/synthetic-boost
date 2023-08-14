#! /bin/bash

# Get first argument from command line
stage=$1

# Check if stage is non empty
if [ -z $stage ]; then
    echo "Please provide a stage as an argument: training or testing"
    exit 1
fi

ext=.png
shape=512

echo "Reshaping image to shape: $shape and saving to $ext format"

python scripts/convert_sitk_to_image.py \
    --image-root /mnt/Enterprise/PUBLIC_DATASETS/camus_database/$stage \
    --glob-pattern 'patient*/patient*_?CH_E?.mhd' \
    --out-dir /mnt/Enterprise/PUBLIC_DATASETS/camus_database_png/$stage/image \
    --out-ext $ext --resample-size $shape

for i in {1..3}; do
    echo "Reshaping masks $i to shape: $shape and saving to $ext format"

    python scripts/convert_sitk_to_image.py \
        --image-root /mnt/Enterprise/PUBLIC_DATASETS/camus_database/$stage \
        --glob-pattern 'patient*/patient*_?CH_E?_gt.mhd' \
        --out-dir /mnt/Enterprise/PUBLIC_DATASETS/camus_database_png/$stage/mask_$i \
        --out-ext $ext --mask-index $i --resample-size $shape
done
