import json
import os

# Make sure we are reading from the right file
input_file = 'baseline_results.jsonl'
output_file = 'hallucination_samples.json'

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found in this directory.")
    exit()

video_scores = {}

# 1. Parse the JSONL file and group by video_id
with open(input_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        vid = data['video_id']
        task = data['task']
        label = str(data['label']).lower().strip()
        prediction = str(data['prediction']).lower().strip()
        
        # Initialize tracking for this video
        if vid not in video_scores:
            video_scores[vid] = {'total': 0, 'correct': 0}
            
        video_scores[vid]['total'] += 1
        
        # 2. Grade the prediction (Skip AV Captioning)
        if task != 'AV Captioning':
            if label in prediction and prediction.startswith(label):
                video_scores[vid]['correct'] += 1

# 3. Filter for the absolute worst performers (0 correct across all tests)
hard_negatives = [vid for vid, scores in video_scores.items() if scores['correct'] == 0]

# 4. Export the list
with open(output_file, 'w') as f:
    json.dump(hard_negatives, f, indent=4)

print(f"Success. Extracted {len(hard_negatives)} hard negative video IDs to {output_file}.")