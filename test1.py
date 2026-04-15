import cv2
import numpy as np
import matplotlib
# 1. 核心步骤：必须在导入 plt 之前，指定使用 'Agg' 后端（专门用于生成文件而非显示窗口）
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 读取图像，转为灰度
img = cv2.imread('facedark.png')
if img is None:
    print("Error: Could not read image.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 直方图均衡化
equ = cv2.equalizeHist(gray)

# 可视化对比
fig, axes = plt.subplots(2, 2, figsize=(10, 8)) # 建议设置一下画布大小
axes[0,0].imshow(gray, cmap='gray')
axes[0,0].set_title('Original Image')
axes[0,1].imshow(equ, cmap='gray')
axes[0,1].set_title('Equalized Image')

axes[1,0].hist(gray.ravel(), 256, [0, 256])
axes[1,0].set_title('Original Histogram')
axes[1,1].hist(equ.ravel(), 256, [0, 256])
axes[1,1].set_title('Equalized Histogram')

# 2. 核心步骤：删除 plt.waitforbuttonpress()，改用 savefig 保存图片
save_path = 'histogram_comparison4.png'
plt.tight_layout() # 自动调整布局防止标题重叠
plt.savefig(save_path, dpi=300) # 保存为高分辨率图片

