import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = "dataset"
CLASSES = ["amman", "kovil", "raja", "murugan"]
IMAGES_PER_CLASS = 5

def create_dummy_dataset():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print(f"Created {DATASET_DIR}")

    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        print(f"Generating images for: {class_name}")
        for i in range(IMAGES_PER_CLASS):
            # Create a random colored image
            img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
            
            # Add text to it so we can visually distinguish (simulating the 'word')
            d = ImageDraw.Draw(img)
            # Just drawing the class name as "text" simulation
            d.text((10,10), class_name, fill=(255, 255, 255))
            
            img.save(os.path.join(class_dir, f"img{i}.jpg"))
            
    print("Done! Dummy dataset created.")
    print("Folder Structure:")
    for root, dirs, files in os.walk(DATASET_DIR):
        level = root.replace(DATASET_DIR, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

if __name__ == "__main__":
    create_dummy_dataset()
