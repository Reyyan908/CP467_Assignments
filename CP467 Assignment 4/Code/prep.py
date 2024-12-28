import os
from PIL import Image
import shutil

# input folder 
input_folder = "crop" 

# output folders
train_folder = "Training Set"
val_folder = "Validation Set"
test_folder = "Test Set"

def preprocess_image(input_path, output_path):
    try:
        img = Image.open(input_path)
        
        # convert to grayscale
        img = img.convert("L")
        
        # resize using Resampling.LANCZOS filter
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        img.save(output_path)
        print(f"Processed and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

# file organizing based off digits 
for digit in range(10):  
    # make sure all digits of proper name and digit are chosen + sorted
    digit_files = [f for f in os.listdir(input_folder) if f.startswith(f"handwritten_{digit}_")]
    digit_files.sort()

    # Select files for training, validation, and testing
    train_files = digit_files[:8] #80 images
    val_files = digit_files[8:10] #20 images
    test_files = digit_files[10:12] #20 images

    # move and rename files for training
    for f in train_files:
        new_name = os.path.splitext(f)[0] + ".jpg"  # rename to .jpg to follow instructions
        input_path = os.path.join(input_folder, f)
        output_path = os.path.join(train_folder, new_name)
        preprocess_image(input_path, output_path)

    # move and rename files for validation
    for f in val_files:
        new_name = os.path.splitext(f)[0] + ".jpg"  # rename to .jpg to follow instructions
        input_path = os.path.join(input_folder, f)
        output_path = os.path.join(val_folder, new_name)
        preprocess_image(input_path, output_path)

    # move and rename files for testing
    for f in test_files:
        new_name = os.path.splitext(f)[0] + ".jpg"  # rename to .jpg to follow instructions
        input_path = os.path.join(input_folder, f)
        output_path = os.path.join(test_folder, new_name)
        preprocess_image(input_path, output_path)

print("Images have been organized into training, validation, and test sets.")