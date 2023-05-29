import numpy as np
import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
folder='/mnt/study/depth/Datasets/kitti_data/2011_09_26/2011_09_26_drive_0005_sync/image_02/data'
folder_2='/mnt/study/depth/Datasets/kitti_data/2011_09_26/2011_09_26_drive_0005_sync/image_00/data'
file_name='0000000000.jpg'
img = cv.imread(os.path.join(folder,file_name))
img_2 = cv.imread(os.path.join(folder_2,file_name))
# img_2 = cv.imread(os.path.join(folder_2,file_name), cv.IMREAD_GRAYSCALE)
# img_2 = img_2[... , np.newaxis]
assert img is not None, "file could not be read, check with os.path.exists()"
# edges = cv.Canny(img,300,300)
hpf = img - cv.GaussianBlur(img, (21,21), 3) +127
hpf_2 = img_2 - cv.GaussianBlur(img_2, (21,21), 1)
# gray = cv.cvtColor(hpf, cv.COLOR_BGR2GRAY)
plt.figure(figsize = (15,2))
plt.subplot(131),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132)#,plt.imshow(edges,cmap = 'gray')
plt.imshow(hpf)
plt.subplot(133),plt.imshow(hpf_2)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()