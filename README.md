# BL_detection — 最大矩形检测算法

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)](https://opencv.org/)

基于 OpenCV 的鲁棒最大矩形检测工具。能够在灰度图像中检测具有显著灰度差异的最大矩形区域，并有效处理光照不均匀和粘连等问题。

---

## 目录

- [功能特性](#功能特性)
- [算法原理](#算法原理)
- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [API 参考](#api-参考)
- [参数调优指南](#参数调优指南)
- [项目结构](#项目结构)
- [许可证](#许可证)

---

## 功能特性

- **自适应阈值分割**：使用高斯自适应阈值处理光照不均匀问题
- **粘连分离**：通过形态学操作（开运算 + 闭运算）有效分离因光照导致的粘连区域
- **多策略回退机制**：
  - 自适应阈值为首选方案
  - Otsu 全局阈值作为第一备选
  - 反转二值图像作为第二备选（处理目标矩形比背景亮的场景）
- **鲁棒矩形识别**：综合矩形度、宽高比、多边形近似顶点数等多重几何验证
- **边缘保持降噪**：使用双边滤波 + 高斯模糊组合进行预处理
- **命令行直接使用**：支持命令行参数输入输出

---

## 算法原理

检测流程分为以下步骤：

```
输入灰度图像
    │
    ▼
┌─────────────────────┐
│ 1. 预处理           │  双边滤波 + 高斯模糊（降噪且保持边缘）
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 2. 自适应二值化     │  高斯自适应阈值（处理光照不均）
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 3. 形态学处理       │  开运算分离粘连 + 闭运算填补缺口
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 4. 轮廓检测与筛选   │  面积过滤 → 矩形度验证 → 宽高比检查 → 选最大
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 5. 回退策略         │  Otsu 阈值 / 反转图像（如主方法未检测到）
└─────────────────────┘
    │
    ▼
输出检测结果（中心坐标、宽高、旋转角度、四角顶点）
```

---

## 环境要求

- Python >= 3.8
- OpenCV >= 4.5.0
- NumPy >= 1.20.0

---

## 安装

```bash
# 克隆仓库
git clone https://github.com/Casit-ARIS-WQL/BL_detection.git
cd BL_detection

# 安装依赖
pip install -r requirements.txt
```

---

## 快速开始

### 命令行使用

```bash
# 基本用法：检测图像中的最大矩形
python detect_largest_rectangle.py <输入图像路径> [输出图像路径]

# 示例
python detect_largest_rectangle.py input.png result.png
```

输出示例：
```
Detected rectangle:
  Center: (320.5, 240.3)
  Size: 200.0 x 150.0
  Angle: -5.2 degrees
  Box corners: [[220, 170], [420, 160], [425, 310], [225, 320]]
Result saved to 'result.png'
```

### 在代码中使用

```python
import cv2
from detect_largest_rectangle import detect_largest_rectangle, draw_detection_result

# 加载灰度图像
img = cv2.imread("input.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测最大矩形
result = detect_largest_rectangle(gray)

if result["rect"] is not None:
    center, size, angle = result["rect"]
    print(f"中心: {center}, 尺寸: {size}, 角度: {angle}")
    print(f"四角顶点: {result['box'].tolist()}")

    # 在图像上绘制结果
    output = img.copy()
    draw_detection_result(output, result, color=(0, 255, 0), thickness=2)
    cv2.imwrite("output.png", output)
else:
    print("未检测到矩形")
```

---

## API 参考

### `detect_largest_rectangle(gray_image, **kwargs)`

主检测函数，检测灰度图像中的最大矩形。

**参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `gray_image` | `np.ndarray` | — | 输入灰度图像（uint8 单通道） |
| `block_size` | `int` | `51` | 自适应阈值的邻域大小（必须为奇数，≥3） |
| `c_offset` | `int` | `10` | 自适应阈值的常数偏移 |
| `morph_ksize` | `int` | `5` | 形态学核大小 |
| `morph_iterations` | `int` | `2` | 形态学操作迭代次数 |
| `min_area_ratio` | `float` | `0.01` | 最小矩形面积占图像面积的比例 |
| `approx_epsilon` | `float` | `0.02` | 轮廓近似容差因子（相对于周长） |
| `use_otsu_fallback` | `bool` | `True` | 是否在自适应方法失败时尝试 Otsu 阈值 |

**返回值：**

返回一个字典，包含：

| 键 | 类型 | 描述 |
|----|------|------|
| `rect` | `tuple` 或 `None` | `((center_x, center_y), (width, height), angle)` |
| `box` | `np.ndarray` 或 `None` | 矩形四个角点坐标（int 类型） |
| `contour` | `np.ndarray` 或 `None` | 检测到的轮廓 |
| `binary` | `np.ndarray` | 最终使用的二值图像 |

**异常：**

- `ValueError`：输入图像为 None、非单通道或非 uint8 类型时抛出

---

### `draw_detection_result(color_image, result, color=(0, 255, 0), thickness=2)`

在彩色图像上绘制检测到的矩形。

**参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `color_image` | `np.ndarray` | — | BGR 彩色图像（会被原地修改） |
| `result` | `dict` | — | `detect_largest_rectangle()` 的返回结果 |
| `color` | `tuple` | `(0, 255, 0)` | 绘制颜色（BGR 格式） |
| `thickness` | `int` | `2` | 线条粗细 |

**返回值：** 绘制后的图像（与输入为同一引用）

---

### `preprocess_image(gray_image, blur_ksize=5)`

预处理函数，使用双边滤波和高斯模糊降噪。

---

### `adaptive_binarize(gray_image, block_size=51, c_offset=10)`

使用高斯自适应阈值进行二值化。

---

### `separate_adhesion(binary_image, morph_ksize=5, iterations=2)`

通过形态学开运算和闭运算分离粘连区域。

---

### `find_largest_rectangle(binary_image, min_area_ratio=0.01, approx_epsilon=0.02)`

在二值图像中查找最大矩形轮廓。

---

## 参数调优指南

| 场景 | 建议调整 |
|------|----------|
| 光照非常不均匀 | 增大 `block_size`（如 101），增大 `c_offset`（如 15） |
| 矩形与背景对比度低 | 减小 `c_offset`（如 5） |
| 矩形边缘有明显粘连 | 增大 `morph_ksize`（如 7）或增加 `morph_iterations`（如 3） |
| 需要检测较小矩形 | 减小 `min_area_ratio`（如 0.005） |
| 矩形形状不规则或有缺损 | 增大 `approx_epsilon`（如 0.04）以放宽近似条件 |
| 图像噪声较大 | 增大预处理中的高斯模糊核 |

---

## 项目结构

```
BL_detection/
├── detect_largest_rectangle.py   # 核心检测算法模块
├── requirements.txt              # Python 依赖
├── .gitignore                    # Git 忽略规则
└── README.md                     # 项目说明文档
```

---

## 许可证

本项目仅供学习和研究使用。如需商业使用，请联系作者获取授权。