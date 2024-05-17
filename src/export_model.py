
import torch

from config import ALL_CLASSES
from model import prepare_model

# Load the checkpoint
checkpoint = torch.load('../outputs/outputs_all-Batch1-28-scheduler/model.pth')

# Set computation device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = prepare_model(num_classes=len(ALL_CLASSES)).to(device)
ckpt = torch.load('../outputs/outputs_all-Batch1-28-scheduler/best_model_iou.pth')
model.load_state_dict(ckpt['model_state_dict'])
model.eval().to(device)

dummy_input = torch.randn(1,3,512,512).to(device)

# Define the output file path for the ONNX model
onnx_model_path = "../outputs/outputs_all-Batch1-28-scheduler/best_model_iou.onnx"

# Export the model
torch.onnx.export(
    model,                    # model being run
    dummy_input,              # model input (or a tuple for multiple inputs)
    onnx_model_path,          # where to save the model (can be a file or file-like object)
    export_params=True,       # store the trained parameter weights inside the model file
    opset_version=11,         # the ONNX version to export the model to
    do_constant_folding=True, # whether to execute constant folding for optimization
    input_names=['input'],    # the model's input names
    output_names=['output'],  # the model's output names
    dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                  'output': {0: 'batch_size'}}
)
