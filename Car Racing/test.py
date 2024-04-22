import os

file_path = 'weights/trial_500.h5'
print("File exists:", os.path.exists(file_path))

try:
    with open(file_path, 'rb') as f:
        print("File can be opened.")
except Exception as e:
    print("Error opening file:", e)