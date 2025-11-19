import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("best_ir8.onnx", providers=['CPUExecutionProvider'])
inp_name = sess.get_inputs()[0].name
# buat dummy input sesuai ukuran yg diharapkan, misal (1,3,224,224)
x = np.random.randn(1,3,224,224).astype(np.float32)
outs = sess.run(None, {inp_name: x})
for i, o in enumerate(outs):
    print(i, np.array(o).shape)
