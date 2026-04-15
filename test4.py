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

# ==========================================
# 核心修改：使用 CLAHE 替代全局直方图均衡化
# ==========================================
# clipLimit: 对比度限制阈值。默认一般是 40，但通常 2.0 到 5.0 之间效果最好，数值越大对比度越强。
# tileGridSize: 图像划分的网格大小。8x8 是最常用的默认值。
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray)
# ==========================================

# 可视化对比
fig, axes = plt.subplots(2, 2, figsize=(12, 10)) # 稍微调大了画布，让图表更清晰
axes[0,0].imshow(gray, cmap='gray')
axes[0,0].set_title('Original Image')
axes[0,0].axis('off') # 关掉坐标轴刻度更好看

axes[0,1].imshow(clahe_img, cmap='gray')
axes[0,1].set_title('CLAHE Image (clipLimit=2.0)')
axes[0,1].axis('off')

axes[1,0].hist(gray.ravel(), 256, [0, 256], color='gray')
axes[1,0].set_title('Original Histogram')

axes[1,1].hist(clahe_img.ravel(), 256, [0, 256], color='gray')
axes[1,1].set_title('CLAHE Histogram')

# 2. 核心步骤：改用 savefig 保存图片
save_path = 'histogram_comparison_clahe.png'
plt.tight_layout() # 自动调整布局防止标题重叠
plt.savefig(save_path, dpi=300) # 保存为高分辨率图片
print(f"✅ 图片已成功保存至: {save_path}")