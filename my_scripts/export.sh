# 设置1张可用的卡

# Deeplabv3p
export CUDA_VISIBLE_DEVICES=0
python export.py \
       --config configs/deeplabv3p/deeplabv3p_resnet50_os8_custom_1024x512_80k.yml \
       --model_path out/train/custom_dataset/deeplabv3p/best_model/model.pdparams \
       --save_dir out/export/deeplabv3p \
       --without_argmax \
       --with_softmax   \
       --input_shape 1 3 720 1280

paddle2onnx --model_dir out/export/deeplabv3p \
        --model_filename model.pdmodel \
        --params_filename model.pdiparams \
        --opset_version 11 \
        --save_file out/export/deeplabv3p/deeplabv3p.onnx


# segformer
python export.py \
       --config configs/segformer/segformer_b2_custom_1024x512_160k.yml \
       --model_path out/train/custom_dataset/segformerB/best_model/model.pdparams \
       --save_dir out/export/segformer \
       --without_argmax \
       --with_softmax   \
       --input_shape 1 3 720 1280

paddle2onnx --model_dir out/export/segformer \
        --model_filename model.pdmodel \
        --params_filename model.pdiparams \
        --opset_version 11 \
        --save_file out/export/segformer/segformer.onnx


# pp-lite-seg
python export.py \
       --config configs/pp_liteseg/pp_liteseg_stdc2_custom_1024x512_scale1.0_160k.yml \
       --model_path out/train/custom_dataset/pp_litesegB/best_model/model.pdparams \
       --save_dir out/export/pp_liteseg \
       --without_argmax \
       --with_softmax   \
       --input_shape 1 3 1080 1920

paddle2onnx --model_dir out/export/pp_liteseg \
        --model_filename model.pdmodel \
        --params_filename model.pdiparams \
        --opset_version 11 \
        --save_file out/export/pp_liteseg/pp_liteseg.onnx


#fcn_hrnetw18   
export CUDA_VISIBLE_DEVICES=0
python export.py \
       --config configs/fcn/fcn_hrnetw18_custom_1024x512_80k.yml  \
       --model_path out/train/custom_dataset/fcn_hrnetw18/best_model/model.pdparams \
       --save_dir out/export/fcn_hrnetw18 \
       --without_argmax \
       --with_softmax \
       --input_shape 1 3 720 1280

export CUDA_VISIBLE_DEVICES=0
paddle2onnx --model_dir out/export/fcn_hrnetw18 \
        --model_filename model.pdmodel \
        --params_filename model.pdiparams \
        --opset_version 11 \
        --save_file out/export/fcn_hrnetw18/fcn_hrnetw18.onnx

python -m paddle2onnx.optimize \
        --input_model out/export/fcn_hrnetw18/fcn_hrnetw18.onnx \
        --output_model out/export/fcn_hrnetw18/fcn_hrnetw18_simple.onnx






#fcn_hrnetw18   
export CUDA_VISIBLE_DEVICES=0
python export.py \
       --config configs/fcn/fcn_hrnetw18_custom_1024x512_80k.yml  \
       --model_path out/train/custom_dataset/fcn_hrnetw18/iter_80000/model.pdparams \
       --save_dir out/export/fcn_hrnetw18 \
       --without_argmax \
       --with_softmax \
       --input_shape 1 3 720 1280

export CUDA_VISIBLE_DEVICES=0
paddle2onnx --model_dir out/export/fcn_hrnetw18 \
        --model_filename model.pdmodel \
        --params_filename model.pdiparams \
        --opset_version 11 \
        --save_file out/export/fcn_hrnetw18/fcn_hrnetw18.onnx