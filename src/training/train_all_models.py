#!/usr/bin/env python3
"""
Comprehensive training script for cardiovascular disease prediction
Trains multiple models on both fundus and ECG datasets
"""

import os
import subprocess
import json
import time
from pathlib import Path

def run_training_command(command, description):
    """Run a training command and return results"""
    print(f"\n{'='*60}")
    print(f"Starting: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            return True
        else:
            print(f"❌ FAILED: {description}")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description} (exceeded 1 hour)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {description} - {str(e)}")
        return False

def main():
    """Main training orchestrator"""
    print("🚀 Starting Comprehensive Cardiovascular Disease Model Training")
    print("This will train multiple models on both fundus and ECG datasets")
    
    # Training configurations
    fundus_configs = [
        {
            "name": "ResNet18",
            "command": 'python train_fundus_cnn.py --data_dir "Fundus_CIMT_2903/Fundus_CIMT_2903 Dataset" --model_type resnet18 --epochs 10 --batch_size 32 --learning_rate 0.001 --pretrained --freeze_backbone',
            "description": "Fundus ResNet18 (pretrained, frozen backbone)"
        },
        {
            "name": "EfficientNet_b0", 
            "command": 'python train_fundus_cnn.py --data_dir "Fundus_CIMT_2903/Fundus_CIMT_2903 Dataset" --model_type efficientnet_b0 --epochs 10 --batch_size 32 --learning_rate 0.001 --pretrained',
            "description": "Fundus EfficientNet B0 (pretrained)"
        },
        {
            "name": "CustomCNN",
            "command": 'python train_fundus_cnn.py --data_dir "Fundus_CIMT_2903/Fundus_CIMT_2903 Dataset" --model_type custom_cnn --epochs 10 --batch_size 32 --learning_rate 0.001',
            "description": "Fundus Custom CNN (from scratch)"
        }
    ]
    
    ecg_configs = [
        {
            "name": "1D_CNN",
            "command": "python train_ecg_models.py --model_type 1d_cnn --epochs 15 --batch_size 64 --learning_rate 0.001 --optimizer adam",
            "description": "ECG 1D CNN"
        },
        {
            "name": "LSTM",
            "command": "python train_ecg_models.py --model_type lstm --epochs 15 --batch_size 32 --learning_rate 0.001 --optimizer adam",
            "description": "ECG LSTM"
        },
        {
            "name": "BiLSTM",
            "command": "python train_ecg_models.py --model_type bilstm --epochs 15 --batch_size 32 --learning_rate 0.001 --optimizer adam",
            "description": "ECG Bidirectional LSTM"
        }
    ]
    
    results = {
        "fundus": {},
        "ecg": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Train Fundus models
    print("\n📸 Starting Fundus Model Training")
    for config in fundus_configs:
        success = run_training_command(config["command"], config["description"])
        results["fundus"][config["name"]] = {
            "status": "success" if success else "failed",
            "description": config["description"]
        }
        time.sleep(2)  # Brief pause between trainings
    
    # Train ECG models
    print("\n📈 Starting ECG Model Training")
    for config in ecg_configs:
        success = run_training_command(config["command"], config["description"])
        results["ecg"][config["name"]] = {
            "status": "success" if success else "failed",
            "description": config["description"]
        }
        time.sleep(2)  # Brief pause between trainings
    
    # Save results
    results_file = "training_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("🏁 TRAINING COMPLETE - SUMMARY")
    print("="*60)
    
    print("\n📸 Fundus Models:")
    for name, result in results["fundus"].items():
        status_emoji = "✅" if result["status"] == "success" else "❌"
        print(f"  {status_emoji} {name}: {result['description']} ({result['status']})")
    
    print("\n📈 ECG Models:")
    for name, result in results["ecg"].items():
        status_emoji = "✅" if result["status"] == "success" else "❌"
        print(f"  {status_emoji} {name}: {result['description']} ({result['status']})")
    
    print(f"\n📊 Results saved to: {results_file}")
    
    # Count successful trainings
    successful_fundus = sum(1 for r in results["fundus"].values() if r["status"] == "success")
    successful_ecg = sum(1 for r in results["ecg"].values() if r["status"] == "success")
    total_successful = successful_fundus + successful_ecg
    total_models = len(fundus_configs) + len(ecg_configs)
    
    print(f"\n📈 Overall Success Rate: {total_successful}/{total_models} models trained successfully")
    
    if total_successful > 0:
        print("\n✨ Next Steps:")
        print("  1. Check model checkpoints in the respective directories")
        print("  2. Evaluate model performance using the evaluation scripts")
        print("  3. Compare results and select best performing models")
        print("  4. Use the models for inference on new data")
    else:
        print("\n⚠️  No models were trained successfully. Please check:")
        print("  - Dataset files are present and accessible")
        print("  - Required libraries are installed")
        print("  - Training scripts are working correctly")

if __name__ == "__main__":
    main()