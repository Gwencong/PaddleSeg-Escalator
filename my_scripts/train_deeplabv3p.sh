#!bin/bash

export CUDA_VISIBLE_DEVICES=1,2 
python -m paddle.distributed.launch train.py \
       --config configs/deeplabv3p/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output

export CUDA_VISIBLE_DEVICES=1,2 
python -m paddle.distributed.launch train.py \
       --config configs/bisenetv1/bisenetv1_resnet18_os8_cityscapes_1024x512_160k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output_bisenet

export CUDA_VISIBLE_DEVICES=1,2
python slim/quant/qat_train.py \
       --config configs/deeplabv3p/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml \
       --model_path output/best_model/model.pdparams \
       --learning_rate 0.001 \
       --do_eval \
       --use_vdl \
       --save_interval 250 \
       --save_dir output_quant


export CUDA_VISIBLE_DEVICES=1,2 
python -m paddle.distributed.launch train.py \
       --config configs/segformer/segformer_b2_cityscapes_1024x512_160k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output_segformer_b2

# HarDNet
export CUDA_VISIBLE_DEVICES=1,2 
python -m paddle.distributed.launch train.py \
       --config configs/hardnet/hardnet_cityscapes_1024x1024_160k.yml \
       --resume_model output_hardnet/iter_93000 \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output_hardnet

# PP-LiteSeg-B2
export CUDA_VISIBLE_DEVICES=1,2 
python -m paddle.distributed.launch train.py \
       --config configs/pp_liteseg/pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_160k.yml \
       --resume_model output_PPliteSeg_B2/iter_154500 \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output_PPliteSeg_B2

visualdl --logdir output_hardnet/ output_PPliteSeg_B2