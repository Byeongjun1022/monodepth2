#%%
import numpy as np
import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
folder='/mnt/study/depth/Datasets/kitti_data/2011_09_26/2011_09_26_drive_0005_sync/image_02/data'
folder_2='/mnt/study/depth/Datasets/kitti_data/2011_09_26/2011_09_26_drive_0005_sync/image_00/data'
file_name='0000000030.jpg'
img = cv.imread(os.path.join(folder,file_name))
img = cv.resize(img, (640,192))
img_2 = cv.imread(os.path.join(folder_2,file_name))
img_2 = cv.resize(img_2, (640,192))
img_3 = cv.imread(os.path.join(folder,file_name), cv.IMREAD_GRAYSCALE)
img_3 = cv.resize(img_3, (640,192))
# img_2 = cv.imread(os.path.join(folder_2,file_name), cv.IMREAD_GRAYSCALE)
# img_2 = img_2[... , np.newaxis]
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img_3 ,300,500)
hpf = img - cv.GaussianBlur(img, (21,21), 3) +127
hpf_2 = img_2 - cv.GaussianBlur(img_2, (21,21), 1)
# gray = cv.cvtColor(hpf, cv.COLOR_BGR2GRAY)
plt.figure(figsize = (30,8))
plt.subplot(231),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232)#,plt.imshow(edges,cmap = 'gray')
plt.imshow(hpf)
plt.subplot(233),plt.imshow(hpf_2)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(edges, cmap='gray')
plt.show()

print(np.sum(edges==255))

# hpf_2 = img_2 - cv.GaussianBlur(img_2, (31,31), 1)
# hpf_list = []
# for i in range(6):
#     hpf_list.append(img_2 - cv.GaussianBlur(img_2, (21,21), i+1))

# plt.figure(figsize = (30,8))
# for i in range(6):
#     plt.subplot(2,3, i+1), plt.imshow(hpf_list[i])
# plt.show()

# added_list = []
# for i in range(6):
#     added_list.append(img_2 + hpf_list[i])

# plt.figure(figsize = (30,8))
# for i in range(6):
#     plt.subplot(2,3, i+1), plt.imshow(added_list[i])
# plt.show()

# %%
