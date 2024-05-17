import numpy as np
import onnx
import onnxruntime as ort
import torch

# Load the ONNX model
onnx_model = onnx.load('/data/projects/IncisionDeepLab/outputs/outputs_all-Batch1-28-scheduler/model.onnx')

# Check the model for any errors
onnx.checker.check_model(onnx_model)

# Run a forward pass with the ONNX Runtime to ensure everything works
ort_session = ort.InferenceSession('/data/projects/IncisionDeepLab/outputs/outputs_all-Batch1-28-scheduler/model.onnx')

# Get the input name for the ONNX model
input_name = ort_session.get_inputs()[0].name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dummy_input = torch.randn(1,3,512,512).to(device)
dummy_input_cpu = dummy_input.cpu()

# Run the model (example with dummy input)
outputs = ort_session.run(None, {input_name: dummy_input_cpu.numpy()})

o = np.array(outputs)
print(o.shape)