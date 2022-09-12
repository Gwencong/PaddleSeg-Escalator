python predict.py \
    --config configs/fcn/fcn_hrnetw18_custom_1024x512_80k.yml \
    --model_path out/train/custom_dataset/fcn_hrnetw18/best_model/model.pdparams \
    --image_path data/custom/test.txt \
    --save_dir out/detect/fcn_hrnetw18