import json
import os

def calculate_accuracy(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    correct = 0
    for item in data:
        gt = str(item['ground_truth']).strip().lower()
        pred = str(item['prediction']).strip().lower()
        
        if gt == pred or gt in pred:
            correct += 1
            
    return (correct / len(data)) * 100 if len(data) > 0 else 0

def main():
    target_dir = r"C:\Users\adars\Downloads"
    
    files = [
        "results_ckpt_acca_only.json",
        "results_ckpt_acca_tati.json",
        "results_ckpt_full_pipeline.json"
    ]
    
    print("\n--- Ablation Table Results ---")
    for file_name in files:
        full_path = os.path.join(target_dir, file_name)
        if os.path.exists(full_path):
            acc = calculate_accuracy(full_path)
            print(f"{file_name.replace('results_', '').replace('.json', '')}: {acc:.2f}%")
        else:
            print(f"Missing file: {file_name}")
    print("------------------------------\n")

if __name__ == "__main__":
    main()