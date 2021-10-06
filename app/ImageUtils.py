import cv2
import numpy as np
from PIL import Image
import os

from app.autoaugment import CIFAR10Policy
from pyspark.sql.types import Row

def augment_image_generator(image_row_data_itr, no_of_augmented_images):

    policy = CIFAR10Policy()
    for image_row_data in image_row_data_itr:

        row_dict = image_row_data.asDict()

        height = row_dict['height']
        width = row_dict['width']
        nChannels = row_dict['nChannels']
        mode = row_dict['mode']

        data = row_dict['data']
        filepath = row_dict['origin']

        filename = os.path.basename(filepath)

        shape = (height, width, nChannels)
        image_np_array = np.ndarray(shape, np.uint8, data)

        pil_image = Image.fromarray(
            cv2.cvtColor(image_np_array, cv2.COLOR_BGR2RGB)
        )

        img_rows = []
        counter = 0

        for _ in range(no_of_augmented_images):
            img = policy(pil_image)
            opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_data = np.asarray(opencvImage.data, dtype=np.uint8)
            length = np.prod(img_data.shape)
            data = bytearray(img_data.astype(dtype=np.int8)[:, :, (2, 1, 0)].reshape(length))

            filename_new = filename+"_"+str(counter)+".JPG"
            filepath_new  = filepath.replace(filename, filename_new)

            img_rows.append(Row(image={'origin': filepath_new, "height": height, "width": width, "nChannels": 3, "mode": mode,
                             "data": data}))
            counter = counter + 1


        yield img_rows


