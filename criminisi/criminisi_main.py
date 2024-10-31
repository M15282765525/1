from inpainter import Inpainter
from skimage.io import imread, imsave
import os

if __name__ == '__main__':
    for i in range(1, 4):
        img_name = os.path.join('testsets','crop', '{}.jpg'.format(i))
        # criminisi mask中白色表示修复区域
        mask_name = os.path.join('testsets', 'crop', 'mask{}.jpg'.format(i))
        inpaint_image = Inpainter(imread(img_name), imread(mask_name, as_gray=True)).inpaint()
        imsave("criminisi{}.jpg".format(i), inpaint_image, quality=100)