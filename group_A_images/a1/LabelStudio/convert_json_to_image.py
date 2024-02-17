"""
Code from https://stackoverflow.com/a/74901690
Convert the exported json from Label Studio to png mask with correct name. the JSON must be located at the same level as the images.
Usage: python convert_json_to_image.py PATH_TO_JSON PATH_TO_OUTPUT_MASKS
Example: python convert_json_to_image.py ../data/raw_data/project-48-at-2024-01-25-08-51-0e3eeda2.json ../data/masks
with the following structure
.
├── data/
│   ├── raw_data/
│   │   ├── PAT_8_15_820.png
│   │   ├── PAT_39_55_233.png
│   │   └── project-48-at-2024-01-25-08-51-0e3eeda2.json
│   └── masks
└── LabelStudio/
    ├── convert_json_to_image.py
    ├── LabelStudio.pdf
    └── merge_masks.py
The folder "masks" will contain the created png version of the masks
"""

from typing import List
import numpy as np
import json
import numpy as np
import sys
import os
from PIL import Image

class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """ from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """ get bit string from bytes data"""
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def rle_to_mask(rle: List[int], height: int, width: int) -> np.array:
    """
    Converts rle to image mask
    Args:
        rle: your long rle
        height: original_height
        width: original_width

    Returns: np.array
    """

    rle_input = InputStream(bytes2bit(rle))

    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = rle_input.read(word_size)
                out[i] = val
                i += 1

    image = np.reshape(out, [height, width, 4])[:, :, 3]
    return image



if __name__ == "__main__":
    json_path = sys.argv[1]
    output_path = sys.argv[2]
    with open(json_path) as f:
        json_ls = json.load(f)

    for _ in json_ls:
        for ann in _["annotations"]:
            result = ann["result"][0]
            filename = _["file_upload"].split("-")[1]
            mask_filename = filename.replace(".png","_mask.png")
            image = rle_to_mask(
                result['value']['rle'], 
                result['original_height'], 
                result['original_width']
            )
            mask_path = os.path.join(output_path,mask_filename)
            im = Image.fromarray(image)
            im.save(mask_path)
            