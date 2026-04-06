# Detect-Defection
Chalk Crack Classifier — a YOLOv8 Nano model that detects crack types in chalk/concrete specimens. Classifies images into 4 categories: bending, tensile, torsional crack, or no crack. Achieves 82.27% top-1 accuracy at just 1.44M parameters. Fast, lightweight, and export-ready. 
Model Overview
PropertyValueArchitectureYOLOv8n-cls (Nano)TaskImage ClassificationNumber of Classes4Input Size224 × 224 pxParameters~1.44 MFile Size~2.97 MBUltralytics Version8.4.14Trained Onchalk_datasetTraining Date2026-02-16

Classes
IDLabel0bending_flexural_crack1no_crack2tensile_mode_I_crack3torsional_helical_crack

Performance
MetricValueTop-1 Accuracy90.27%Top-5 Accuracy100.00%Validation Loss0.4617Fitness Score0.9513

Best checkpoint was saved at epoch 10 (top-1 accuracy: 90.27%).


Training Configuration
HyperparameterValueEpochs50Batch Size16Image Size224Base Modelyolov8n-cls.pt (pretrained)OptimizerAutoLearning Rate (lr0)0.01LR Final (lrf)0.01Weight Decay0.0005Warmup Epochs3AugmentationRandAugmentDropout0.0AMP (Mixed Precision)✅ EnabledWorkers8Seed0

Quick Start
Installation
bashpip install ultralytics
Inference (Python)
pythonfrom ultralytics import YOLO

# Load the model
model = YOLO("best.pt")

# Run inference on an image
results = model("path/to/your/image.jpg")

# Print the predicted class
print(results[0].probs.top1)          # Class index
print(results[0].names[results[0].probs.top1])  # Class name
Inference (CLI)
bashyolo classify predict model=best.pt source=path/to/your/image.jpg
Batch Inference on a Folder
pythonfrom ultralytics import YOLO

model = YOLO("best.pt")
results = model("path/to/image/folder/")

for r in results:
    cls_name = r.names[r.probs.top1]
    confidence = r.probs.top1conf.item()
    print(f"{r.path}: {cls_name} ({confidence:.2%})")

Export
The model can be exported to multiple formats for deployment:
pythonfrom ultralytics import YOLO

model = YOLO("best.pt")

model.export(format="onnx")        # ONNX (cross-platform)
model.export(format="torchscript") # TorchScript
model.export(format="tflite")      # TensorFlow Lite (mobile)
model.export(format="coreml")      # Core ML (iOS/macOS)

File Structure (Expected Dataset Layout)
chalk_dataset/
├── train/
│   ├── bending_flexural_crack/
│   ├── no_crack/
│   ├── tensile_mode_I_crack/
│   └── torsional_helical_crack/
└── val/
    ├── bending_flexural_crack/
    ├── no_crack/
    ├── tensile_mode_I_crack/
    └── torsional_helical_crack/

Re-training / Fine-tuning
pythonfrom ultralytics import YOLO

model = YOLO("best.pt")  # start from this checkpoint

model.train(
    data="path/to/chalk_dataset",
    epochs=30,
    imgsz=224,
    batch=16,
    lr0=0.001,       # lower LR when fine-tuning
)

Requirements
ultralytics>=8.4.0
torch>=2.0.0
torchvision
pillow

Notes

This is the best checkpoint (best.pt), saved at peak validation accuracy during training — not the final epoch weights.
The model was trained with pretrained ImageNet weights (yolov8n-cls.pt) and then fine-tuned, which is why convergence was fast.
Top-5 accuracy is 100% across all 50 epochs, meaning the correct class was always in the model's top-4 predictions — strong signal that the feature space is well-separated.
If deploying on edge devices, consider exporting to TFLite or ONNX for reduced latency.


License
Ultralytics YOLOv8 is released under the AGPL-3.0 License.
Model weights trained on proprietary data are subject to the data owner's terms.
