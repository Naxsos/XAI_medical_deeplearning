import os
import shutil
from PIL import Image
import subprocess
import sys

# The problematic file
problem_file = 'resized_images_2/00004673_016.png'
backup_file = problem_file + '.bak'

# Make a backup of the original file
if os.path.exists(problem_file):
    print(f"Creating backup of {problem_file} to {backup_file}")
    shutil.copy2(problem_file, backup_file)

# Get file information
file_size = os.path.getsize(problem_file)
print(f"File size: {file_size} bytes")

# Try to repair using ImageMagick (if available)
try:
    print("\nAttempting repair with ImageMagick...")
    result = subprocess.run(['convert', problem_file, '-strip', problem_file + '.fixed.png'], 
                           capture_output=True, text=True)
    if result.returncode == 0:
        print("ImageMagick repair successful!")
        # Replace the original with the fixed one
        shutil.move(problem_file + '.fixed.png', problem_file)
    else:
        print(f"ImageMagick repair failed: {result.stderr}")
except FileNotFoundError:
    print("ImageMagick not installed. Skipping this repair method.")

# Try other approaches
print("\nAttempting alternate repair methods...")

try:
    # Method 1: Try to open and resave with PIL
    print("Method 1: Open and resave with PIL...")
    try:
        img = Image.open(problem_file)
        img.save(problem_file + '.pil.png')
        print("Method 1 successful!")
        shutil.move(problem_file + '.pil.png', problem_file)
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Try with OpenCV
    print("\nMethod 2: Using OpenCV...")
    try:
        import cv2
        img = cv2.imread(problem_file, cv2.IMREAD_UNCHANGED)
        if img is not None:
            cv2.imwrite(problem_file + '.cv2.png', img)
            print("Method 2 successful!")
            shutil.move(problem_file + '.cv2.png', problem_file)
        else:
            print("Method 2 failed: OpenCV couldn't read the image")
    except ImportError:
        print("Method 2 skipped: OpenCV not available")
    
    # Method 3: Try converting through different format
    print("\nMethod 3: Converting through BMP format...")
    try:
        subprocess.run(['convert', backup_file, 'bmp:-', '|', 'convert', 'bmp:-', problem_file], 
                      shell=True, check=True)
        print("Method 3 successful!")
    except Exception as e:
        print(f"Method 3 failed: {e}")
    
    # Final test - can we open it now?
    print("\nFinal verification...")
    try:
        img = Image.open(problem_file)
        print(f"SUCCESS! Image can now be opened with PIL")
        print(f"Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
    except Exception as e:
        print(f"Image is still corrupted: {e}")
        print("\nRecommendations:")
        print("1. Remove this image from your dataset")
        print("2. Create a new empty image with the same name")
        print("3. Check if there are more corrupted images with similar issues")
        
        # If all repair attempts failed, restore the backup
        if os.path.exists(backup_file):
            print("\nRestoring original file from backup...")
            shutil.copy2(backup_file, problem_file)
            
except Exception as e:
    print(f"Repair process error: {e}") 