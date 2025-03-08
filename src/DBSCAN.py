import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    return binary


def draw_lane_points(image, points):
    for point in points:
        cv2.circle(image, point, 10, (0, 0, 255), -1)
    return image

# 生成一些示例车道线点数据
# 假设这些点是从图像中提取的
def find_lane_points(binary_image, num_points=500):
    height, width = binary_image.shape
    # start = height // 2
    start = 1
    end = height - 1
    row_indices = np.linspace(start, end, num_points, dtype=int)

    left_lane_points = []
    right_lane_points = []

    for row in row_indices:
        row_data = binary_image[row]
        mid_point = width // 2

        # Find left lane point
        left_point = next((i for i in range(mid_point, 0, -1) if row_data[i] == 0), None)
        if left_point:
            left_lane_points.append((left_point, row))

        # Find right lane point
        right_point = next((i for i in range(mid_point, width) if row_data[i] == 0), None)
        if right_point:
            right_lane_points.append((right_point, row))

    lane_points = np.array(left_lane_points + right_lane_points)

    return left_lane_points, right_lane_points, lane_points


def detect_DBSCAN(image):
    # 使用DBSCAN聚类算法
    width = image.shape[1]
    binary_image = process_image(image)

    left_points, right_points, points = find_lane_points(binary_image)
    if points.ndim == 2:
        db = DBSCAN(eps=100, min_samples=2).fit(points)
        labels = db.labels_

        points_dict = group_samples_by_label(points, labels)
        points_dict = filter_samples_by_percentage(points_dict)
        points_dict = categorize_lane_lines(points_dict, width)

        image = draw_lanes(image, points_dict)
        direction = decide_direction(points_dict, width)
    else:
        direction = 'FORWARD'
    return image, direction


def group_samples_by_label(samples, labels):
    """
    将样本按标签分组

    参数:
    samples (list): 样本列表
    labels (list): 标签列表

    返回:
    dict: 按标签分组的样本字典
    """
    grouped_samples = {}

    # 遍历标签和样本，将样本按标签分组
    for label, sample in zip(labels, samples):
        if label not in grouped_samples:
            grouped_samples[label] = []
        grouped_samples[label].append(sample)

    return grouped_samples


def filter_samples_by_percentage(grouped_samples, threshold=0.2):
    """
    过滤掉样本数占比低于阈值的键

    参数:
    grouped_samples (dict): 按标签分组的样本字典
    threshold (float): 样本数占比的阈值，默认为0.2

    返回:
    dict: 过滤后的样本字典
    """
    # 计算总样本数
    total_samples = sum(len(samples) for samples in grouped_samples.values())

    # 过滤字典，只保留样本数占比超过阈值的键
    filtered_samples = {label: np.array(samples) for label, samples in grouped_samples.items()
                        if len(samples) / total_samples > threshold}

    return filtered_samples


def categorize_lane_lines(samples_by_label, image_width):
    """
    将样本根据其最下方的点的位置分类为左车道线或右车道线

    参数:
    samples_by_label (dict): 按标签分组的样本字典，样本是点坐标 (x, y) 的列表
    image_width (int): 图像的宽度

    返回:
    dict: 包含左车道线和右车道线的样本字典
    """
    categorize_samples = {'left': np.array([[-80, 0]]),
                          'right': np.array([[400, 0]])
                          }

    for label, samples in samples_by_label.items():
        # 找到y坐标最大的点
        lowest_point = max(samples, key=lambda point: point[1])

        # 判断这个点位于左半平面还是右半平面
        if lowest_point[0] < image_width / 2:
            categorize_samples['left'] = samples
        else:
            categorize_samples['right'] = samples
    return categorize_samples


def draw_lanes(image, categorize_samples):
    colors = {'left': (255, 0, 0), 'right': (0, 255, 0)}

    # 绘制聚类结果
    for key in categorize_samples.keys():
        xy = categorize_samples[key]
        col = colors[key]
        for x, y in xy:
            cv2.circle(image, (int(x), int(y)), 2, (int(col[0] * 255), int(col[1] * 255), int(col[2] * 255)), -1)
        center = np.mean(xy, axis=0).astype(int)

        cv2.putText(image, key, center, cv2.FONT_HERSHEY_SIMPLEX, 2, col, 4)
    return image


def decide_direction(categorize_samples, image_width):
    centers = {}
    extremes = {}
    for label, xy in categorize_samples.items():
        centers[label] = np.mean(xy, axis=0).astype(int)
        lowest_point = np.array(min(xy, key=lambda point: point[1]))
        extremes[label] = lowest_point
    if centers:
        centers = np.array(list(centers.values()))
        center = np.mean(centers, axis=0)
    if extremes:
        extremes = np.array(list(extremes.values()))
        extreme = np.mean(extremes, axis=0)

        judge = extreme

        threshold = [0.05, 0.4, 0.6, 0.95]

        if image_width * threshold[2] < judge[0] <= image_width * threshold[3]:
            direction = 'FORWARD_RIGHT'
        elif image_width * threshold[0] <= judge[0] <= image_width * threshold[1]:
            direction = 'FORWARD_LEFT'
        elif image_width * threshold[1] <= judge[0] < image_width * threshold[2]:
            direction = 'FORWARD'
        elif image_width * threshold[3] < judge[0]:
            direction = 'RIGHT'
        elif judge[0] < image_width * threshold[0]:
            direction = 'LEFT'
        else:
            direction = 'STOP'
    else:
        direction = 'STOP'
    return direction


if __name__ == '__main__':
    # 示例数据
    samples = ['sample0', 'sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6', 'sample7', 'sample8',
               'sample9']
    labels = [0, 1, 0, 2, 1, 2, 0, 1, 2, 0]

    # 调用函数
    grouped_samples = group_samples_by_label(samples, labels)

    # 打印分组结果
    for label, group in grouped_samples.items():
        print(f"Label {label}: {group}")

    points = np.array([
        [100, 300], [105, 305], [110, 310], [115, 315], [120, 320],  # 左车道点
        [400, 300], [405, 305], [410, 310], [415, 315], [420, 320],  # 右车道点
        [200, 400], [205, 405], [210, 410], [215, 415], [220, 420],  # 噪声点
    ])
    # 创建一个空白图像用于绘制结果
    height, width = 500, 600
    black = np.zeros((height, width, 3), dtype=np.uint8)
    output_image = detect_DBSCAN(black, points)

    cv2.imshow('output_image', output_image)
    cv2.waitKey(0)
