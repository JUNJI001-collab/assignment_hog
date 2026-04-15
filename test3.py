# 服务器无图形界面配置
import matplotlib

matplotlib.use('Agg')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# 读取图片
# ─────────────────────────────────────────────
img_path = "facedark.png"
img_bgr = cv2.imread(img_path)

if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 实例化 CLAHE 对象
# clipLimit: 颜色对比度的阈值，超过该阈值的部分会被裁剪并均匀分布到其他部分
# tileGridSize: 图像划分的网格大小，默认是 8x8
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ═══════════════════════════════════════════════════════════
# 方法1：RGB三通道分别应用 CLAHE
# ═══════════════════════════════════════════════════════════
def apply_clahe_rgb(image_rgb):
    channels = cv2.split(image_rgb)
    # 分别对R, G, B通道应用CLAHE
    eq_channels = [clahe.apply(ch) for ch in channels]
    return cv2.merge(eq_channels)


result_method1 = apply_clahe_rgb(img_rgb)


# ═══════════════════════════════════════════════════════════
# 方法2：YCrCb亮度通道应用 CLAHE
# ═══════════════════════════════════════════════════════════
def apply_clahe_ycrcb(image_rgb):
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # 仅对亮度通道Y应用CLAHE
    y_eq = clahe.apply(y)

    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    result_bgr = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


result_method2 = apply_clahe_ycrcb(img_rgb)


# ─────────────────────────────────────────────
# 评价指标：信息熵 & 标准差
# ─────────────────────────────────────────────
def entropy(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist[hist > 0] / hist.sum()
    return -np.sum(hist * np.log2(hist))


def std_dev(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return gray.std()


metrics = {
    "Original": (entropy(img_rgb), std_dev(img_rgb)),
    "Method1 (RGB)": (entropy(result_method1), std_dev(result_method1)),
    "Method2 (YCrCb)": (entropy(result_method2), std_dev(result_method2)),
}

print("=" * 45)
print(f"{'':15s} {'Entropy':>8s}  {'Std':>8s}")
print("-" * 45)
for name, (e, s) in metrics.items():
    print(f"{name:15s} {e:8.4f}  {s:8.4f}")
print("=" * 45)

# ─────────────────────────────────────────────
# 清爽白底可视化（论文友好）
# ─────────────────────────────────────────────
titles = [
    "Original",
    "Method1: RGB CLAHE\n(Color Distortion)",
    "Method2: YCrCb CLAHE\n(Preserve Color)"
]
images = [img_rgb, result_method1, result_method2]

# 白底画布
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.3)

for col, (title, img) in enumerate(zip(titles, images)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 行1：显示图像
    ax_img = fig.add_subplot(gs[0, col])
    ax_img.imshow(img)
    ax_img.set_title(title, color="black", fontsize=11, pad=8)
    ax_img.axis("off")
    e, s = metrics[list(metrics.keys())[col]]
    ax_img.set_xlabel(f"Entropy={e:.3f}  σ={s:.2f}", color="black", fontsize=9)

    # 行2：灰度直方图
    ax_gray = fig.add_subplot(gs[1, col])
    ax_gray.set_facecolor("white")
    hist_g = cv2.calcHist([gray], [0], None, [256], [0, 256])
    ax_gray.fill_between(range(256), hist_g.ravel(), color="gray", alpha=0.5)
    ax_gray.plot(hist_g.ravel(), color="black", linewidth=1)
    ax_gray.set_xlim([0, 255])
    ax_gray.set_title("Grayscale Histogram", color="black", fontsize=9)
    ax_gray.tick_params(colors="black", labelsize=7)

    # 行3：通道直方图
    ax_ch = fig.add_subplot(gs[2, col])
    ax_ch.set_facecolor("white")

    if col == 0:
        # 原图RGB
        for i, ccolor in enumerate(["red", "green", "blue"]):
            ch = img[:, :, i]
            hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
            ax_ch.plot(hist.ravel(), color=ccolor, linewidth=1, label=["R", "G", "B"][i])
        ax_ch.legend(fontsize=8)
        ax_ch.set_title("R/G/B Channels", color="black", fontsize=9)

    elif col == 1:
        # RGB CLAHE
        for i, ccolor in enumerate(["red", "green", "blue"]):
            ch = img[:, :, i]
            hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
            ax_ch.plot(hist.ravel(), color=ccolor, linewidth=1, label=["R(clahe)", "G(clahe)", "B(clahe)"][i])
        ax_ch.legend(fontsize=8)
        ax_ch.set_title("CLAHE R/G/B", color="black", fontsize=9)

    else:
        # Y通道对比
        img_bgr_orig = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        y_orig = cv2.split(cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2YCrCb))[0]
        img_bgr_eq = cv2.cvtColor(result_method2, cv2.COLOR_RGB2BGR)
        y_eq = cv2.split(cv2.cvtColor(img_bgr_eq, cv2.COLOR_BGR2YCrCb))[0]

        hist_y_orig = cv2.calcHist([y_orig], [0], None, [256], [0, 256])
        hist_y_eq = cv2.calcHist([y_eq], [0], None, [256], [0, 256])
        ax_ch.plot(hist_y_orig.ravel(), color="orange", linewidth=1, label="Y Original")
        ax_ch.plot(hist_y_eq.ravel(), color="green", linewidth=1, label="Y CLAHE")
        ax_ch.legend(fontsize=8)
        ax_ch.set_title("Y Channel Comparison", color="black", fontsize=9)

    ax_ch.set_xlim([0, 255])
    ax_ch.tick_params(colors="black", labelsize=7)

# 总标题
fig.suptitle("Color CLAHE Comparison | RGB vs YCrCb",
             color="black", fontsize=14, y=0.98)

plt.savefig("clahe_color_result.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print("\nClean result saved to: clahe_color_result.png")