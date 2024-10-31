import cv2
import numpy as np
from skimage import morphology


img = cv2.imread('fixed_image.jpg', cv2.IMREAD_GRAYSCALE)

_, binary_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

binary_otsu = morphology.binary_opening(binary_otsu, selem=morphology.disk(3))
import numpy as np
from scipy import ndimage
from skimage import morphology
binary_otsu = morphology.binary_opening(binary_otsu, selem=morphology.disk(3))
from skimage import io, filters, segmentation, color, exposure
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
from skimage.morphology import disk, binary_opening
image_path = 'fixed_image.jpg'  
image = io.imread(image_path, as_gray=True)
otsu_threshold = filters.threshold_otsu(image)
binary_otsu = image > otsu_threshold
binary_otsu = binary_otsu.astype(np.uint8)
binary_otsu = morphology.binary_opening(binary_otsu, selem=morphology.disk(3)) 


distance = ndimage.distance_transform_edt(binary_otsu)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=binary_otsu) 
markers = ndimage.label(local_maxi)[0]
labels_ws = segmentation.watershed(-distance, markers, mask=binary_otsu)


colored_labels = color.label2rgb(labels_ws, bg_label=0)

otsu_threshold = filters.threshold_otsu(colored_labels)
binary_otsu = image > otsu_threshold


binary_otsu = binary_otsu.astype(np.uint8)
binary_otsu = morphology.binary_opening(binary_otsu, selem=morphology.disk(3)) 


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(binary_otsu, cmap='gray')
ax[1].set_title('Otsu Thresholding')
ax[2].imshow(colored_labels)
ax[2].set_title('Watershed Segmentation')

cv2.imwrite('Otsu_Thresholding.jpg', (binary_otsu * 255).astype(np.uint8))
plt.tight_layout()
plt.show()