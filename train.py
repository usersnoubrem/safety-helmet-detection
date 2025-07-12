from ultralytics import YOLO
import yaml
import os

def train_model():
    """
    训练YOLOv8安全帽检测模型
    """
    # 1. 加载预训练模型
    model = YOLO('yolov8n.pt')  # 使用YOLOv8n预训练模型

    # 2. 配置训练参数
    train_args = {
        'data': 'data/data.yaml',
        'epochs': 50,  # 训练轮数
        'imgsz': 640,  # 图像大小
        'batch': 16,  # 批次大小
        'optimizer': 'AdamW',  # 优化器
        'lr0': 0.001,  # 初始学习率
        'name': 'helmet_detection_v1',  # 实验名称
        'augment': True,  # 数据增强
        'patience': 10,  # 早停耐心值
        'save': True,  # 保存模型
        'device': 0  # 使用GPU（如果有）
    }

    # 3. 开始训练
    results = model.train(**train_args)

    # 4. 保存最佳模型
    best_model_path = 'models/best.pt'
    model.save(best_model_path)
    print(f"训练完成! 最佳模型保存至: {best_model_path}")

    model.export(
        format='onnx',
        imgsz=640,
        dynamic=True,  # 动态批处理支持
        simplify=True,  # 简化ONNX图
        opset=12  # ONNX算子集版本
    )
    # 5. 模型评估
    metrics = model.val()
    print(f"评估结果: mAP@0.5 = {metrics.box.map50:.4f}, mAP@0.5:0.95 = {metrics.box.map:.4f}")

    return model


if __name__ == '__main__':
    # 确保数据目录存在
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 创建数据集配置文件（示例）
    data_config = {
        'train': 'data/images/train',
        'val': 'data/images/val',
        'names': ['helmet', 'no_helmet', 'person']  # 类别名称
    }

    with open('data/data.yaml', 'w') as f:
        yaml.dump(data_config, f)

    print("开始训练安全帽检测模型...")
    train_model()