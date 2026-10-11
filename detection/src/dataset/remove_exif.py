import os
from typing import List, Tuple

from PIL import Image, ImageOps


def remove_exif(
        img_path_list: List[str],
        replace_folder: Tuple[str, str] = ("/raw/", "/remove_exif/")
    ):
    """load image from **/raw/**, then remove exif and save to **/remove_exif/**"""
    assert all(replace_folder[0] in img_path for img_path in img_path_list)    
    for img_path in img_path_list:
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)
        output_dir = img_path.replace(replace_folder[0], replace_folder[1])
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        img.save(output_dir)
        