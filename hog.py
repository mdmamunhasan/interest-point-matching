import os
from skimage import io
from skimage import img_as_float
from skimage import exposure
from skimage import feature
import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Input Image 1
filename = os.path.join("images", 'cv_cover1.jpg')
img = io.imread(filename)
img = img_as_float(img)

(H, hogImage) = feature.hog(
    img, orientations=9, pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
    visualize=True
)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")

plt.imshow(hogImage)
plt.show()
