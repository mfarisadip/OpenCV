import onnx
from onnx import helper, numpy_helper

def convert_ir_version(input_path, output_path, target_ir_version=8):
    """Convert ONNX model to different IR version"""
    # Load the model
    model = onnx.load(input_path)

    print(f"Original model IR version: {model.ir_version}")
    print(f"Original model opset version: {model.opset_import[0].version}")

    # Change IR version
    model.ir_version = target_ir_version

    # Try to save the model
    try:
        onnx.save(model, output_path)
        print(f"Successfully converted to IR version {target_ir_version}: {output_path}")

        # Verify the model
        loaded_model = onnx.load(output_path)
        print(f"Verification - New IR version: {loaded_model.ir_version}")
        print(f"Verification - Opset version: {loaded_model.opset_import[0].version}")

        return True
    except Exception as e:
        print(f"Failed to convert: {e}")
        return False

# Try converting the existing model to IR version 8
print("Attempting to convert existing ONNX model to IR version 8...")
input_model = "/Users/sweetjichu/Documents/faris-lab/Roots/runs/detect/train3/weights/best.onnx"
output_model = "best_ir_version_8.onnx"

success = convert_ir_version(input_model, output_model, target_ir_version=8)

if not success:
    print("\nTrying IR version 7...")
    output_model = "best_ir_version_7.onnx"
    convert_ir_version(input_model, output_model, target_ir_version=7)

    print("\nTrying IR version 6...")
    output_model = "best_ir_version_6.onnx"
    convert_ir_version(input_model, output_model, target_ir_version=6)