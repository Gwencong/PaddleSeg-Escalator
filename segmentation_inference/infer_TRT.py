import os
import cv2
import onnx
import numpy as np
import onnxruntime
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from PIL import Image
from time import time
from pathlib import Path

colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255],[127,64,127],[0, 0, 0]]
classes = ['left baffle','right baffle','step','floor plate','background']
train_id_to_color = np.array(colors)
train_id_to_color = np.ascontiguousarray(train_id_to_color)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRT_Infer():
    def __init__(self,engine_path,shape=[1,3,640,640],num_classes=4) -> None:
        self.shape = shape              # 模型的输入图片形状
        self.num_classes = num_classes  # 类别数量
        self.logger = trt.Logger(trt.Logger.WARNING)    # tensorrt日志记录器
        self.runtime = trt.Runtime(self.logger)         # tensorrt运行时
        self.engine = self.load_engine(engine_path,self.runtime)    # 导入TRT模型
        self.context = self.engine.create_execution_context()       # 获取执行上下文
        self.stream = cuda.Stream()                                 # 获取数据处理流
        self.inputs, self.outputs, self.bindings = self.allocate_buffers_dynamic(self.context,self.shape)
    
    def normalize(self, im, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
        # 图片归一化
        mean = np.array(mean)[np.newaxis, np.newaxis, :]    # [1,1,3]
        std = np.array(std)[np.newaxis, np.newaxis, :]      # [1,1,3]
        im = im.astype(np.float32, copy=False) / 255.0
        im -= mean
        im /= std
        return im

    def load_engine(self,engine_path,runtime):
        # 加载TRT模型
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine

    def allocate_buffers_dynamic(self,context,shape):
        # 分配device内存
        inputs = []
        outputs = []
        bindings = []
        context.set_binding_shape(0,shape)   # Dynamic Shape 模式需要绑定真实数据形状
        engine = context.engine
        for binding in engine:
            ind = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(ind)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings

    def preprocess(self, im, pagelocked_buffer):
        # 预处理, 并将预处理结果拷贝到分配的主机内存上
        img = self.normalize(im)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = np.expand_dims(img, 0)
        img = np.asarray(img,dtype=np.float32)
        np.copyto(pagelocked_buffer, (img.astype(trt.nptype(trt.float32)).ravel()))
        return img

    def inference(self,im):
        # preprocess
        img = self.preprocess(im, self.inputs[0].host)
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        model_out = self.outputs[0].host
        # postprocess
        mask,color_mask = self.postprocess(model_out, img.shape)
        return mask,color_mask
    
    def postprocess(self,model_out,shape):
        # 后处理, argmax获取每个像素的预测类别
        b,c,h,w = shape
        pred = model_out.reshape(b,self.num_classes,h,w)[0]
        pred = np.argmax(pred,axis=0).astype(np.int64)          # [H,W]
        color_pred = train_id_to_color[pred].astype(np.uint8)   # [H,W,3]
        return pred, color_pred


def get_contour_approx(pred,img,visual=False):
    '''根据预测的mask获取扶梯左右挡板、梯路的轮廓\n
    Args:
        pred: 预测的mask, 尺寸: [H,W], 每个像素的值为0-3, 对于类别id
        img: 原图, 可视化用
    Return: 
        approxs: 获取到的轮廓点集, list, 有三个元素, 对应左右挡板和梯路的区域轮廓
    '''
    h,w = pred.shape[:2]
    approxs = []
    for i in range(3):
        mask = np.where(pred==i,0,255).astype(np.uint8)
        contours,hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(contour) for contour in contours]
        indexes = [j for j,area in enumerate(areas) if 0.01<area/(h*w)<0.8]
        contour = contours[indexes[0]]
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        # approx = cv2.convexHull(contour)
        approxs.append(approx)
        if visual:
            cv2.drawContours(img,[approx],-1,(0,255,255))
    if visual:
        cv2.imwrite('out-aprroxs.jpg',img)
    return approxs


def infer_trt(img_path,model_path):
    img = cv2.imread(img_path)
    input_shape = [1,3,img.shape[0],img.shape[1]]
    model = TRT_Infer(engine_path=model_path,shape=input_shape)
    mask,color_mask = model.inference(img)
    
    img = cv2.addWeighted(img,0.7,color_mask,0.3,0)
    cv2.imwrite('out_trt_mask.jpg',color_mask)
    cv2.imwrite('out_trt_fuse.jpg',img)


if __name__ == "__main__":
    img_path = 'test.jpg'
    model_path = 'out/export/pp_liteseg/pp_liteseg.trt'
    infer_trt(img_path,model_path)
    