import glob
import os
import cv2
import numpy as np
from pathlib import Path


class Dataset:
    # dataloader for tflite int 8.
    def __init__(self, img_size=(640, 640)):
        self.img_size = img_size

    def data_load(self):
        root = Path(__file__).resolve().parent
        path = root / 'coco128/images/train2017'
        files = None
        if path.exists():
            if path.is_dir():
                files = sorted(glob.glob(os.path.join(path, '*.*')))
            elif path.is_file():
                files = [path]
        else:
            raise Exception(f'ERROR: {path} does not exist! please place '
                            f'the coco128 as introduced in README.md')

        images = [x for x in files if x.split('.')[-1].lower() == 'jpg']
        dataset = []
        for file in images:
            img = cv2.imread(file)
            img = self._pad(img, self.img_size)
            dataset.append(img)

        return dataset

    def _pad(self, img, input_size):
        # Padding image to expected size.
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

        return padded_img


def representative_dataset_gen(dataset):
    # Representative dataset generator for use with converter.representative_dataset.
    for n, img in enumerate(dataset):
        input = np.expand_dims(img, axis=0).astype(np.float32)
        input /= 255
        yield [input]
        if n >= 100:
            break
