import os
import shutil
import random

# Source directory
src = "C:/Python/kiwi/datasets/croissant"
# Destination directories
train_dir = "C:/Python/kiwi/datasets/train/croissant"
valid_dir = "C:/Python/kiwi/datasets/valid/croissant"
test_dir = "C:/Python/kiwi/datasets/test/croissant"

# Ensure destination directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List all files in the source directory
data = os.listdir(src)

# Shuffle the data to ensure randomness
random.shuffle(data)

# Calculate split sizes
total_files = len(data)
train_split = int(0.8 * total_files)
valid_split = int(0.15 * total_files)

# Split the data
train_files = data[:train_split]
valid_files = data[train_split:train_split + valid_split]
test_files = data[train_split + valid_split:]

# Function to move files to respective directories
def move_files(file_list, destination):
    for file_name in file_list:
        src_path = os.path.join(src, file_name)
        dest_path = os.path.join(destination, file_name)
        shutil.copy(src_path, dest_path)

# Move the files
move_files(train_files, train_dir)
move_files(valid_files, valid_dir)
move_files(test_files, test_dir)

print(f"Data split completed:")
print(f"{len(train_files)} files moved to {train_dir}")
print(f"{len(valid_files)} files moved to {valid_dir}")
print(f"{len(test_files)} files moved to {test_dir}")
