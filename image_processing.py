import numpy as np
from PIL import Image
import tensorflow as tf


class ImageProcessing:
    def __init__(self):
        pass

    def image_to_array(self, input_img, img_width, img_height):
        # img = Image.open(input_img)
        img = input_img.resize((img_width, img_height), resample=0, box=None)
        # img = ImageOps.grayscale(img)
        img = img.convert('RGB')
        img = np.array(img)
        img = img / 255.0
        img = tf.image.rgb_to_grayscale(img)
        if img_width > 32:
            #img = tf.image.rgb_to_grayscale(img)
            return img[np.newaxis]
        return img

    def separate_chars(self, input_img, chars=[], return_img=False):
        #if not chars:
            #input_img = Image.open(input_img)

        width = input_img.width - 256

        if width > 255:
            img = input_img.copy()
            img = img.crop((256, 0, img.width, 256))
            cropped = input_img.crop((0, 0, 256, 256))
            chars.append(cropped)

            if return_img:
                return chars[0]

            return self.separate_chars(img, chars)

        chars.append(input_img)
        print(f'Number of characters: {len(chars)}')
        a_img = np.array([self.image_to_array(i, 32, 32) for i in chars])
        return a_img

