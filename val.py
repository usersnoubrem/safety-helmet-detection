from ultralytics import YOLO

def main():
    # 加载已训练的模型
    model = YOLO('models/best.pt')  # 这里假设最佳模型权重路径是这个

    # 评估模型
    results = model.val(
        data='data.yaml',  # 与训练时使用的数据集配置文件相同
        imgsz=640,         # 图像尺寸，与训练时保持一致
        batch=16,          # 批次大小，可根据实际情况调整
        device=0           # GPU设备ID (0表示第一块GPU)
    )

if __name__ == '__main__':
    # Windows 多进程需要的额外处理（可选，但建议加上）
    import multiprocessing
    multiprocessing.freeze_support()  # 解决冻结打包问题，非打包时也可兼容

    # 执行主函数
    main()