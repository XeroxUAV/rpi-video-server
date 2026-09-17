import numpy as np
import onnxruntime as ort


MODEL_PATH = "onnx-models/best_1.onnx"
IMAGE_PATH = "test.jpg"

# -------------------------
# Load model
# -------------------------

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
)

input_info = session.get_inputs()[0]
input_name = input_info.name

print("Input name:", input_name)
print("Input shape:", input_info.shape)
print("Input type:", input_info.type)

for output in session.get_outputs():
    print(
        "Output:",
        output.name,
        output.shape,
        output.type,
    )

