import onnxruntime as ort

print("Available ONNX Runtime providers:")
print(ort.get_available_providers())

if "CUDAExecutionProvider" in ort.get_available_providers():
    print("\nGPU SUCCESS")
    print("CUDAExecutionProvider is available.")
else:
    print("\nGPU NOT AVAILABLE")
    print("InsightFace will fall back to CPU.")