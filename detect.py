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
from PIL import Image
from pathlib import Path

from postprocess import get_contour_approx


colors = [(255, 0, 0),(0, 255, 0),(0, 0, 255),(0, 0, 0)]
train_id_to_color = np.array(colors)
train_id_to_color = np.ascontiguousarray(train_id_to_color)

def file_size(path):
    # Return file/dir size (MB)
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def normalize(im, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im

def postprocess(model_out):
    target = model_out[0][0]
    target[target == 255] = 19
    return train_id_to_color[target].astype(np.uint8)

def export_trt():
    prefix = colorstr('TensorRT')
    print(f'\n{prefix} starting export with TensorRT {trt.__version__}...')

    verbose = True
    workspace = 4
    half = True
    try:
        onnx_path = "out/export/fcn_hrnetw18/fcn_hrnetw18_simple.onnx"
        trt_path = "out/export/fcn_hrnetw18/fcn_hrnetw18.trt"
        
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            raise RuntimeError(f'failed to load ONNX file: {onnx_path}')

        profile = builder.create_optimization_profile()     
        profile.set_shape("x", (1, 3, 640, 640), (1, 3, 720, 1280), (1, 3, 1080, 1920))
        config.add_optimization_profile(profile)

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        print(f'{prefix} Network Description:')
        for inp in inputs:
            print(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
        for out in outputs:
            print(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')

        print(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine in {trt_path}')
        if builder.platform_has_fast_fp16 and half:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, open(trt_path, 'wb') as t:
            t.write(engine.serialize())
        print(f'{prefix} export success, saved as {trt_path} ({file_size(trt_path):.1f} MB)')
        return trt_path
    except Exception as e:
        print(f'\n{prefix} export failure: {e}')

def infer_onnx(img_path):
    onnx_path = "out/export/fcn_hrnetw18/fcn_hrnetw18.onnx"
    # img_path = 'data/custom_mini/images/20220727_160039236.jpg'
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
    Image.fromarray(pred).save('out.jpg')

def infer_trt(img_path):
    img = cv2.imread(img_path)
    # img = Image.open(img_path).convert('RGB')
    # img = img.resize((640,640))
    # input_shape = [1,3,img.height,img.width]
    input_shape = [1,3,img.shape[0],img.shape[1]]
    model = TRT_Infer(engine_path='out/export/fcn_hrnetw18/fcn_hrnetw18.trt',shape=input_shape)
    for i in range(5):
        t = time()
        out = model.do_inference_v2(img)
        pred = model.postprocess(out,has_argmax=False)
        print(f'{(time()-t)*1000:.2f}ms')
    pred[0].save(os.path.join('out/detect', img_path.split('/')[-1].split('.')[0]+'_trt.png'))
    pred[1].save(os.path.join('out/detect', img_path.split('/')[-1].split('.')[0]+'_trt_added.png'))
    get_contour_approx(pred[2],img)

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
        self.shape = shape
        self.num_classes = num_classes
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path,self.runtime)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers_dynamic(self.context,self.shape)
        self.img = None
        

    def load_engine(self,engine_path,runtime):
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine

    def allocate_buffers_dynamic(self,context,shape):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
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
        return inputs, outputs, bindings, stream

    def preprocess(self,im, pagelocked_buffer):
        self.img = im
        img = normalize(im)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = np.expand_dims(img, 0)
        img = np.asarray(img,dtype=np.float32)
        np.copyto(pagelocked_buffer, (img.astype(trt.nptype(trt.float32)).ravel()))
        return img,im

    def do_inference_v2(self,img):
        self.preprocess(img,self.inputs[0].host)
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in self.outputs]
    
    def postprocess(self,model_out,has_argmax=False):
        b,c,h,w = self.shape
        if has_argmax:
            preds = model_out[0].reshape(b,h,w)[0]
        else:
            preds = model_out[0].reshape(b,self.num_classes,h,w)
            preds = np.argmax(preds,axis=1)[0]
        preds = np.asarray(preds,dtype=np.int64)
        # preds[preds == 255] = 0
        colorized_preds = train_id_to_color[preds].astype(np.uint8)
        added_img = cv2.addWeighted(self.img,0.7,colorized_preds,0.3,0)
        colorized_preds = Image.fromarray(colorized_preds)
        added_img = Image.fromarray(added_img)
        return colorized_preds,added_img,preds


if __name__ == "__main__":
    img_path = 'data/custom_mini/images/20220727_160039236.jpg'
    # export_trt()
    # infer_onnx(img_path)
    infer_trt(img_path)
    

