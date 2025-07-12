import cv2
import time
from ultralytics import YOLO
from utils.helpers import plot_detections, check_safety_violation

def real_time_detection(model_path):
    """
    实时安全帽检测
    """
    # 1. 加载训练好的模型
    model = YOLO(model_path)

    # 2. 打开摄像头
    cap = cv2.VideoCapture(0)  # 0表示默认摄像头

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 3. 获取摄像头参数
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"摄像头分辨率: {frame_width}x{frame_height}, FPS: {fps}")

    # 4. 定义类别名称和颜色
    class_names = model.names
    colors = [
        (0, 255, 0),  # helmet - 绿色
        (0, 0, 255),  # no_helmet - 红色
        (255, 255, 0)  # person - 黄色
    ]

    # 5. 性能统计
    frame_count = 0
    start_time = time.time()
    violation_detected = False
    last_violation_time = 0

    print("开始实时检测... (按 'q' 键退出)")

    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            break

        # 6. 执行推理
        results = model(frame, verbose=False)  # 禁用详细输出

        # 7. 处理检测结果
        boxes = []
        scores = []
        class_ids = []

        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

        # 8. 检查安全违规
        current_time = time.time()
        if check_safety_violation(class_ids, class_names):
            violation_detected = True
            last_violation_time = current_time
        elif current_time - last_violation_time > 2:  # 2秒后重置违规状态
            violation_detected = False

        # 9. 在帧上绘制结果
        if len(boxes) > 0:
            frame = plot_detections(frame, boxes, scores, class_ids, class_names, colors)

        # 10. 显示安全状态
        if violation_detected:
            cv2.putText(frame, "SAFETY VIOLATION: NO HELMET!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 11. 显示帧率
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 12. 显示结果
        cv2.imshow("Safety Helmet Detection", frame)

        # 13. 退出条件
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 14. 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print("检测结束")


if __name__ == '__main__':
    # 使用训练好的模型
    model_path = "models/best.pt"

    # 运行实时检测
    real_time_detection(model_path)