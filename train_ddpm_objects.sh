directories=../trained_models

for dir in "$directories"/*; do
    dir=$(basename ${dir})
    echo "$dir"
    python3 train_diffusion_policy.py -o $dir -t train_ddpm -g 1 -d ../trained_models
done
