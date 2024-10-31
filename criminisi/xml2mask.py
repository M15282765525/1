import os.path

import numpy as np
from lxml import etree
from PIL import Image


def load_img(img_path):
    return Image.open(img_path)


def save_img(img_np, img_path):
    Image.fromarray(img_np).save(img_path)


if __name__ == '__main__':
    for i in range(1, 4):
        xml_file = os.path.join('testsets', 'ori', '{}.xml'.format(i))
        img = load_img(os.path.join('testsets','ori', '{}.jpg'.format(i)))
        mask_img = np.zeros_like(img)
        # 设置初始值
        mask_img[:, :, :] = 255
        tree = etree.parse(xml_file)
        root = tree.getroot()

        objects = root.findall('object')
        for obj in objects:
            name = obj.find('name').text
            difficult = obj.find('difficult').text
            bndbox = obj.find('bndbox')
            bbox = [
                int(float(bndbox.find('xmin').text)),
                int(float(bndbox.find('ymin').text)),
                int(float(bndbox.find('xmax').text)),
                int(float(bndbox.find('ymax').text))
            ]
            # 设置破损区域值
            mask_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = 0
        save_img(mask_img, os.path.join('testsets', 'ori', 'mask{}_reverse.jpg'.format(i)))