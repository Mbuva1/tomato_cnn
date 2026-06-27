# Save as check_dataset.py and run: python check_dataset.py

import os

dataset_path = 'dataset'

if not os.path.exists(dataset_path):
    print(f"❌ Dataset folder '{dataset_path}' not found!")
    print(f"   Current directory: {os.getcwd()}")
else:
    print(f"✅ Dataset folder found: {dataset_path}")
    print("\nFolders found:")
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
            print(f"  📁 {folder}: {len(files)} images")
            if files:
                print(f"     First file: {files[0]}")