# fcn_hrnetw18
export CUDA_VISIBLE_DEVICES=1,2 
python -m paddle.distributed.launch train.py \
       --config configs/fcn/fcn_hrnetw18_custom_1024x512_80k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 100 \
       --save_dir out/train/custom_dataset/fcn_hrnetw18 \
       --resume_model out/train/custom_dataset/fcn_hrnetw18/iter_80000

export CUDA_VISIBLE_DEVICES=0 
python -m paddle.distributed.launch train.py \
       --config configs/fcn/fcn_hrnetw18_custom_1024x512_80k.yml \
       --use_vdl \
       --save_interval 100 \
       --save_dir out/train/custom_dataset/fcn_hrnetw18_8_80000 
# finetune
export CUDA_VISIBLE_DEVICES=2
python -m paddle.distributed.launch train.py \
       --config configs/fcn/fcn_hrnetw18_custom_1024x512_80k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 100 \
       --save_dir out/train/custom_dataset/fcn_hrnetw18_class_weight \
       --resume_model out/train/custom_dataset/fcn_hrnetw18/iter_146000

# deeplabv3p
export CUDA_VISIBLE_DEVICES=2 
python -m paddle.distributed.launch train.py \
       --config configs/deeplabv3p/deeplabv3p_resnet50_os8_custom_1024x512_80k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 100 \
       --save_dir out/train/custom_dataset/deeplabv3p

# PP-Lite-Seg-B
export CUDA_VISIBLE_DEVICES=0 
python -m paddle.distributed.launch train.py \
       --config configs/pp_liteseg/pp_liteseg_stdc2_custom_1024x512_scale1.0_160k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 100 \
       --save_dir out/train/custom_dataset/pp_litesegB
       --resume_model out/train/custom_dataset/pp_litesegB/iter_44100

# segformer-B
export CUDA_VISIBLE_DEVICES=0,2 
python -m paddle.distributed.launch train.py \
       --config configs/segformer/segformer_b2_custom_1024x512_160k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 100 \
       --save_dir out/train/custom_dataset/segformerB