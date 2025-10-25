#!/usr/bin/env python3
"""
Create labels.csv and metadata.json files for fundus dataset
"""

import json
import pandas as pd
from pathlib import Path
import glob

def create_fundus_files():
    """Create labels.csv and metadata.json from data_info.json"""
    
    # Load data_info.json
    with open('data_info.json', 'r') as f:
        data_info = json.load(f)
    
    # Create labels.csv
    labels_data = []
    metadata_data = {}
    
    # Process each subject
    for subject_id, subject_data in data_info.items():
        label = subject_data['label']
        
        # Add entries for both eyes
        if 'right_eye' in subject_data:
            right_filename = subject_data['right_eye']
            labels_data.append({
                'filename': right_filename.replace('.png', ''),
                'label': label,
                'eye': 'right',
                'subject_id': subject_id
            })
        
        if 'left_eye' in subject_data:
            left_filename = subject_data['left_eye']
            labels_data.append({
                'filename': left_filename.replace('.png', ''),
                'label': label,
                'eye': 'left',
                'subject_id': subject_id
            })
        
        # Add to metadata
        metadata_data[subject_id] = {
            'gender': subject_data['gender'],
            'thickness': subject_data['thickness'],
            'label': label,
            'group': subject_data['group'],
            'true_age': subject_data['True_age'],
            'age': subject_data['age'],
            'right_eye': subject_data.get('right_eye', ''),
            'left_eye': subject_data.get('left_eye', '')
        }
    
    # Create labels.csv
    labels_df = pd.DataFrame(labels_data)
    labels_df.to_csv('Fundus_CIMT_2903/Fundus_CIMT_2903 Dataset/labels.csv', index=False)
    print(f"Created labels.csv with {len(labels_data)} entries")
    
    # Create metadata.json
    with open('Fundus_CIMT_2903/Fundus_CIMT_2903 Dataset/metadata.json', 'w') as f:
        json.dump(metadata_data, f, indent=2)
    print(f"Created metadata.json with {len(metadata_data)} subjects")
    
    # Verify images exist
    image_dir = Path('Fundus_CIMT_2903/Fundus_CIMT_2903 Dataset/images')
    missing_images = []
    
    for entry in labels_data:
        filename = entry['filename'] + '.png'
        if not (image_dir / filename).exists():
            missing_images.append(filename)
    
    if missing_images:
        print(f"Warning: {len(missing_images)} images not found:")
        for img in missing_images[:10]:  # Show first 10
            print(f"  - {img}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
    else:
        print("All images verified successfully!")

if __name__ == "__main__":
    create_fundus_files()