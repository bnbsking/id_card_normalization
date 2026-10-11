import glob
import os

from src.dataset.remove_exif import remove_exif


if __name__ == "__main__":
    img_path_list = glob.glob("/dataset/images/raw/0914/*/*.jpg") \
        + glob.glob("/dataset/images/raw/0914/*/*.jpeg")
    black = {"20129549876037.jpg"}
    
    img_path_list = [img_path for img_path in img_path_list if os.path.basename(img_path) not in black]
    remove_exif(img_path_list)
