import os
import shutil
import random
from PIL import Image
import numpy as np


dataset_path = r"E:\3phasessolutions\design document solution\Designdoc\WasteDataset\waste_datasets\waste_datasets\orignal_datasets"

train_path = os.path.join(dataset_path, "train")
test_path  = os.path.join(dataset_path, "test")


SPLIT_RATIO  = 0.7        # 70% train, 30% test
SEED         = 42
IMG_SIZE     = (224, 224) 

random.seed(SEED)

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path,  exist_ok=True)


def preprocess_and_save(src_path, dst_path):
    """
    Loads an image, resizes to 224x224, normalizes pixels to 0-1,
    and saves the processed image to the destination path.
    """
    try:
        img = Image.open(src_path).convert("RGB")  
        img = img.resize(IMG_SIZE, Image.LANCZOS)   

     
        img_array = np.array(img, dtype=np.float32) / 255.0

      
        img_saved = Image.fromarray((img_array * 255).astype(np.uint8))
        img_saved.save(dst_path)
        return True

    except Exception as e:
        print(f"  ⚠️  Skipped {os.path.basename(src_path)}: {e}")
        return False


print(f"{'Category':<20} {'Train':>6} {'Test':>6} {'Total':>6} {'Skipped':>8}")
print("-" * 52)

total_train = total_test = total_skipped = 0

for category in sorted(os.listdir(dataset_path)):
    category_path = os.path.join(dataset_path, category)

  
    if not os.path.isdir(category_path) or category in ["train", "test"]:
        continue

   
    all_images = [
        f for f in os.listdir(category_path)
        if os.path.isfile(os.path.join(category_path, f))
        and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]

    random.shuffle(all_images)

    split_point = int(len(all_images) * SPLIT_RATIO)
    train_imgs  = all_images[:split_point]
    test_imgs   = all_images[split_point:]

    train_cat_path = os.path.join(train_path, category)
    test_cat_path  = os.path.join(test_path,  category)
    os.makedirs(train_cat_path, exist_ok=True)
    os.makedirs(test_cat_path,  exist_ok=True)

    skipped = 0


    for img in train_imgs:
        src = os.path.join(category_path, img)
        dst = os.path.join(train_cat_path, img)
        if not preprocess_and_save(src, dst):
            skipped += 1

  
    for img in test_imgs:
        src = os.path.join(category_path, img)
        dst = os.path.join(test_cat_path, img)
        if not preprocess_and_save(src, dst):
            skipped += 1

    saved_train = len(train_imgs) - skipped
    print(f"{category:<20} {len(train_imgs):>6} {len(test_imgs):>6} {len(all_images):>6} {skipped:>8}")
    total_train   += len(train_imgs)
    total_test    += len(test_imgs)
    total_skipped += skipped

print("-" * 52)
print(f"{'TOTAL':<20} {total_train:>6} {total_test:>6} {total_train+total_test:>6} {total_skipped:>8}")
print("\n✅ Preprocessing + Dataset split complete!")