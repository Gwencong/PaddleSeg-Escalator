export CUDA_VISIBLE_DEVICES=0 
python val.py \
    --config configs/fcn/fcn_hrnetw18_custom_1024x512_80k.yml \
    --model_path out/train/custom_dataset/fcn_hrnetw18/best_model/model.pdparams \
    
    --aug_eval \
    --scales 0.75 1.0 1.25 \
    --flip_vertical \

    --is_slide \
    --crop_size 256 256 \
    --stride 128 128

export CUDA_VISIBLE_DEVICES=0 
python val.py \
    --config configs/fcn/fcn_hrnetw18_custom_1024x512_80k.yml \
    --model_path out/train/custom_dataset/fcn_hrnetw18/iter_80000/model.pdparams \