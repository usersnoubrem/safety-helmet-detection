import cv2
import numpy as np


def plot_detections(image, boxes, scores, class_ids, class_names, colors):
    """
    在图像上绘制检测结果
    """
    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    # 绘制每个检测框
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        color = colors[class_id]

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 创建标签文本
        label = f"{class_names[class_id]}: {score:.2f}"

        # 计算文本大小
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_size, text_thickness
        )

        # 绘制文本背景
        cv2.rectangle(
            image,
            (x1, y1 - label_height - baseline),
            (x1 + label_width, y1),
            color,
            -1
        )

        # 绘制文本
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 255),
            text_thickness
        )

    return image


def check_safety_violation(class_ids, class_names):
    """
    检查安全违规情况（未戴安全帽）
    """
    # 安全违规的类别（根据实际数据集调整）
    unsafe_classes = ["no_helmet", "no_hat"]

    # 检查是否有违规类别
    for class_id in class_ids:
        if class_names[class_id] in unsafe_classes:
            return True
    return False