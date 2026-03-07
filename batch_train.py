import time
import sys
import os
import traceback
import argparse
from fixed_distillation_trainer import FixedDistillationTrainer

# Define the models to train
# We prioritize using the 'cans' specific models if they exist, otherwise fallback to standard.
# Based on user request order: v8 -> v11 -> v26
# Define the models to train
# We prioritize using the 'cans' specific models if they exist, otherwise fallback to standard.
# Based on user request order: v8 -> v11 -> v26
models_to_train = [
    {
        "name": "yolov8n_cans",
        "file": "project/models/teachers/yolov8_teacher.pt",
        "fallback": "yolov8n.pt",
        "force_rebuild": True  # Force merge of all datasets before first run
    },
    {
        "name": "yolo11n_cans",
        "file": "project/models/teachers/yolov11_teacher.pt",
        "fallback": "yolo11n.pt"
    },
    {
        "name": "yolo26n_cans",
        "file": "project/models/teachers/yolov26_teacher.pt",
        "fallback": "yolo26n.pt" # Fresh blank student architecture (distilled from scratch)
    }
]

# Training configuration
EPOCHS = 150
BATCH_SIZE = 8

# Dynamically resolve paths
import os
import pathlib

# Since Colab places this somewhere absolute vs Windows local...
_possible_paths = [
    "/content/CS-228-Project/project/datasets/model_training_data/data.yaml", 
    "/content/CS-228-Project/datasets/model_training_data/data.yaml",
    "/content/drive/MyDrive/CS-228_Project/project/datasets/model_training_data/data.yaml",
    str(pathlib.Path(__file__).resolve().parent / "datasets" / "model_training_data" / "data.yaml")
]

DATA_CONFIG = None
for _p in _possible_paths:
    if os.path.exists(_p):
        DATA_CONFIG = _p
        break

if not DATA_CONFIG:
    # Just default to relative and pray!
    DATA_CONFIG = "datasets/model_training_data/data.yaml"

FRACTION = 0.1  # Train on 1/3 of the dataset to speed up training

def run_training_step(model_info, batch_size=BATCH_SIZE, epochs=EPOCHS, fraction=FRACTION, device=None, cache=False):
    model_name = model_info["name"]
    model_file = model_info["file"]
    
    # Check if file exists, else use fallback
    if not os.path.exists(model_file):
        print(f"Warning: {model_file} not found. Checking fallback {model_info.get('fallback')}...")
        fallback = model_info.get("fallback")
        if fallback and (os.path.exists(fallback) or not fallback.startswith("models/")):
             # Fallback exists OR it's a standard ultralytics name (like yolov8n.pt) which auto-downloads
             model_file = fallback
        else:
             print(f"Error: No valid model file found for {model_name}. Skipping.")
             return False

    student_name = f"student_{model_name}"
    print(f"\n{'='*60}")
    print(f"Starting Training for: {model_name}")
    print(f"Teacher: {model_file}")
    print(f"Output Name: {student_name}")
    print(f"{'='*60}\n")
    
    start_t = time.time()
    try:
        # Initialize and run via Python instead of subprocess to avoid command injection
        # According to the project proposal, Teacher Models (finetuned on Cans with Blur) 
        # should distill their knowledge down to Nano Student Models (default untrained architectures)
        
        # We determine the raw student arch from the fallback configuration
        # For example: yolov8n.pt, yolo11n.pt, or yolo26n.pt depending on the teacher's structure
        target_student_architecture = model_info["fallback"]
        
        trainer = FixedDistillationTrainer(
            teacher_name=model_file,
            student_name=target_student_architecture,
            data_cfg=DATA_CONFIG,
            epochs=epochs,
            batch_size=batch_size,
            run_name=student_name,
            fraction=fraction,
            device=device,
            cache=True
        )
        trainer.train()
        success = True
    except Exception as e:
        print(f"\n❌ {model_name} training failed with exception:")
        traceback.print_exc()
        success = False
        
    end_t = time.time()
    
    duration = (end_t - start_t) / 60
    if success:
        print(f"\n✅ {model_name} training completed successfully in {duration:.1f} minutes.")
        best_model_path = f"project/{student_name}_best.pt"
        print(f"   Best model saved to: {best_model_path}")
        
        # Post-training evaluation on Golden Test Set & Latency Validation
        golden_test_set_yaml = "datasets/golden_test_set/data.yaml"
        print(f"\n--- Running Evaluation for {student_name} ---")
        try:
            from ultralytics import YOLO
            import torch
            
            # Load the newly trained student model
            eval_model = YOLO(best_model_path)
            
            # 1. Validation on Golden Test Set (if exists)
            if os.path.exists(golden_test_set_yaml):
                print(f"Golden Test Set found at {golden_test_set_yaml}. Running validation...")
                metrics = eval_model.val(data=golden_test_set_yaml, split='test')
                print(f"Golden Test Set mAP50: {metrics.box.map50:.4f}")
            else:
                print(f"Golden Test Set NOT found at {golden_test_set_yaml}. Skipping test set validation.")
                print("Note: Team is currently curating the golden test set.")
                
            # 2. Latency Benchmarking
            print("Running latency benchmark...")
            eval_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            eval_model.to(eval_device)
            dummy_input = torch.zeros(1, 3, 640, 640).to(eval_device)
            # Warmup
            for _ in range(10):
                _ = eval_model(dummy_input, verbose=False)
            
            # Measure latency
            start_infer = time.time()
            iters = 50
            for _ in range(iters):
                _ = eval_model(dummy_input, verbose=False)
            end_infer = time.time()
            
            avg_latency_ms = ((end_infer - start_infer) / iters) * 1000
            print(f"✅ {student_name} Average Inference Latency: {avg_latency_ms:.2f} ms")
            
        except Exception as e:
            print(f"⚠ Evaluation failed for {student_name}:")
            traceback.print_exc()
            
        return True
    else:
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch Train Student Models")
    parser.add_argument("--model", type=str, default="all", choices=["all", "yolov8", "yolov11", "yolov26"], help="Which model to train")
    parser.add_argument("--teacher", type=str, default=None, help="Explicit path to a teacher model to use (only use if training a single model type).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size for training (default: {BATCH_SIZE}).")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Epochs for training (default: {EPOCHS}).")
    parser.add_argument("--fraction", type=float, default=FRACTION, help=f"Fraction of dataset to use (default: {FRACTION}).")
    parser.add_argument("--device", type=str, default=None, help="Device to use for training (e.g. '0' or 'cpu').")
    args = parser.parse_args()

    selected_models = []
    for m in models_to_train:
        base_model = dict(m)  # copy to avoid mutating global
        if args.teacher:
            base_model["file"] = args.teacher
            
        if args.model == "all":
            selected_models.append(base_model)
        elif args.model == "yolov8" and "8n" in base_model["name"]:
            selected_models.append(base_model)
        elif args.model == "yolov11" and "11n" in base_model["name"]:
            selected_models.append(base_model)
        elif args.model == "yolov26" and "26n" in base_model["name"]:
            selected_models.append(base_model)

    if not selected_models:
        print(f"No models matched selection: {args.model}")
        return

    print(f"🚀 Starting Batch Training Sequence for: {[m['name'] for m in selected_models]}")
    if args.teacher:
        print(f"🧑‍🏫 Overriding teacher model to explicit path: {args.teacher}")
        
    results = {}
    
    for model_info in selected_models:
        success = run_training_step(
            model_info, 
            batch_size=args.batch_size,
            epochs=args.epochs,
            fraction=args.fraction,
            device=args.device
        )
        results[model_info["name"]] = "Success" if success else "Failed"
        
        # Optional: Sleep briefly between runs to let GPU cool?
        if success:
            time.sleep(10)

    print("\n" + "="*50)
    print("BATCH TRAINING SUMMARY")
    print("="*50)
    for name, status in results.items():
        print(f"{name}: {status}")

if __name__ == "__main__":
    main()
