import onnxruntime
import paddle
import onnx
import numpy as np
import paddle.nn.functional as F
from PIL import Image
from time import time
import cv2

colors = [(128, 64, 128),(244, 35, 232),(70, 70, 70),(102, 102, 156),(190, 153, 153),(153, 153, 153),
        (250, 170, 30),(220, 220, 0),(107, 142, 35),(152, 251, 152),(70, 130, 180),(220, 20, 60),
        (255, 0, 0),(0, 0, 142),(0, 0, 70),(0, 60, 100),(0, 80, 100),(0, 0, 230),(119, 11, 32),[0,0,0]]
train_id_to_color = np.array(colors)
train_id_to_color = np.ascontiguousarray(train_id_to_color)

def normalize(im, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
    mean = mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im

def postprocess(model_out):
    target = model_out[0][0]
    target[target == 255] = 19
    return train_id_to_color[target].astype(np.uint8)

def infer_onnx():
    onnx_path = "output_onnx_fcn/new_model.onnx"
    img_path = '/home/gwc/gwc/code/DeepLabV3Plus-Pytorch/samples/berlin_000000_000019_leftImg8bit.png'
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_path,providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    im = cv2.imread(img_path)
    # im = cv2.resize(im,(1080,768))
    img = normalize(im)
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = np.expand_dims(img, 0)
    img = np.asarray(img,dtype=np.float32)

    print("input shape:", img.shape)
    ort_inputs = {ort_session.get_inputs()[0].name: img}

    t_list = []
    for i in range(20):
        t1 = time()
        ort_output = ort_session.run(None, ort_inputs)
        pred = postprocess(ort_output)
        cost = (time()-t1)*1000
        print(f'{cost:.2f}ms')
        t_list.append(cost)
    print(sum(t_list)/len(t_list),'ms')
    # cv2.imwrite('out_onnx.jpg',pred)
    Image.fromarray(pred).save('out_onnx_segformer.jpg')

if __name__ == "__main__":
    infer_onnx()