import onnx
from onnx import version_converter, checker

# Ganti dengan nama file ONNX kamu
model = onnx.load("best_ir8.onnx")

# Tampilkan IR version dan opset version
print("IR version:", model.ir_version)
print("Opset imports:", [(op.domain, op.version) for op in model.opset_import])
print("Producer:", model.producer_name, model.producer_version)

# # convert opset ke 8
# converted = version_converter.convert_version(model, 8)

# # cek validitas model hasil konversi
# checker.check_model(converted)

# # simpan
# onnx.save(converted, "model_opset8.onnx")
# print("Saved model_opset8.onnx")


# import onnx
# from onnx import checker

# # Load model dengan IR version 8
# model_path = "/Users/sweetjichu/Documents/faris-lab/Roots/best_ir_version_8.onnx"
# model = onnx.load(model_path)

# # Tampilkan informasi model
# print("Model IR version:", model.ir_version)
# print("Model opset imports:")
# for opset_import in model.opset_import:
#     print(f"  Domain: {opset_import.domain or 'ai.onnx'}, Version: {opset_import.version}")

# # Validasi model
# try:
#     checker.check_model(model)
#     print("Model is valid!")

#     # Simpan ulang untuk memastikan konsistensi
#     onnx.save(model, "model_ir8_final.onnx")
#     print("Model saved as model_ir8_final.onnx")

# except Exception as e:
#     print(f"Model validation failed: {e}")
