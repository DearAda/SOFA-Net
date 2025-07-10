import os
import cv2
import matplotlib.pyplot as plt
import random


def generate_colors(num_colors):
    """
    生成 num_colors 个不同的颜色
    """
    random.seed(42)  # 保持颜色一致性
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_colors)]


def visualize_yolo_labels(image_folder, label_folder=None, class_names=None):
    """
    :param image_folder: 图片所在文件夹路径
    :param label_folder: 标签所在文件夹路径，如果与图片文件夹相同可设置为 None
    :param class_names:  类别列表(可选)，用于显示类别名称，否则只显示 class_id
    """
    if label_folder is None:
        label_folder = image_folder  # 如果标签和图片在同一文件夹

    # 获取文件夹下所有图片文件名
    valid_exts = [".jpg", ".png", ".jpeg"]
    image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f.lower())[-1] in valid_exts]

    # 生成类别颜色映射
    num_classes = len(class_names) if class_names else 80  # 默认 80 类，适用于 COCO
    class_colors = generate_colors(num_classes)

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        txt_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(label_folder, txt_file)

        if not os.path.exists(label_path):
            print(f"标签文件不存在，跳过: {label_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取图像，跳过: {img_path}")
            continue

        height, width = image.shape[:2]

        # 读取标签
        bboxes = []
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    print(f"标签格式有误，跳过这一行: {line}")
                    continue

                class_id, x_center, y_center, w, h = parts
                class_id = int(class_id)
                x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)

                x1 = int((x_center - w / 2) * width)
                y1 = int((y_center - h / 2) * height)
                x2 = int((x_center + w / 2) * width)
                y2 = int((y_center + h / 2) * height)

                bboxes.append((class_id, x1, y1, x2, y2))

        # 画框和标签
        for bbox in bboxes:
            class_id, x1, y1, x2, y2 = bbox
            color = class_colors[class_id % num_classes]  # 确保索引不超出范围

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            if class_names is not None and 0 <= class_id < len(class_names):
                label_str = class_names[class_id]
            else:
                label_str = str(class_id)

            cv2.putText(image, label_str, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 显示图片
        plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Image: {img_file}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    class_names = None  # 例如: ["person", "car", "dog"]

    visualize_yolo_labels(
        image_folder=r"D:\Desktop\VisDrone_Datasets\train\images",
        label_folder=r"D:\Desktop\VisDrone_Datasets\train\labels",
        class_names=class_names
    )