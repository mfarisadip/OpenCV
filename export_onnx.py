from ultralytics import YOLO

# Load the PyTorch model
model = YOLO('/Users/sweetjichu/Documents/faris-lab/Roots/runs/detect/train3/weights/best.pt')

# Export to ONNX with older opset for IR version 8 compatibility
print("Exporting to ONNX with opset 7 (for IR version 8 compatibility)...")
try:
    model.export(
        format='onnx',
        opset=7,   # Use opset 7 for IR version 8
        imgsz=640,  # Image size
        batch=1,    # Batch size
        dynamic=False,  # Keep it simple
        simplify=False   # Don't simplify for now
    )
    print("Successfully exported model_opset7.onnx")
except Exception as e:
    print(f"Failed to export with opset 7: {e}")

    # Try opset 8 as fallback
    print("Trying with opset 8...")
    try:
        model.export(
            format='onnx',
            opset=8,
            imgsz=640,
            batch=1,
            dynamic=False,
            simplify=False
        )
        print("Successfully exported model_opset8.onnx")
    except Exception as e2:
        print(f"Failed to export with opset 8: {e2}")

        # Try opset 9 as final fallback
        print("Trying with opset 9...")
        try:
            model.export(
                format='onnx',
                opset=9,
                imgsz=640,
                batch=1,
                dynamic=False,
                simplify=False
            )
            print("Successfully exported model_opset9.onnx")
        except Exception as e3:
            print(f"Failed to export with opset 9: {e3}")

# Check existing ONNX file properties
import onnx
try:
    existing_model = onnx.load("/Users/sweetjichu/Documents/faris-lab/Roots/runs/detect/train3/weights/best.onnx")
    print("\nExisting model info:")
    print(f"IR version: {existing_model.ir_version}")
    print(f"Opset version: {existing_model.opset_import[0].version}")
except Exception as e:
    print(f"Could not load existing model: {e}")