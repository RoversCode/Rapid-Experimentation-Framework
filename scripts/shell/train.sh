
export CUDA_VISIBLE_DEVICES=2,3,4,5,7

exp_name="test_base"
torchrun \
    --nproc_per_node=5 \
    train.py \
    --exp_name $exp_name

