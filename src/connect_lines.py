import cv2
import numpy as np
from DBSCAN import process_image


path_image = '../images/20240704112131~1.jpg'
# 读取二值化图像
image = cv2.imread(path_image)
image = process_image(image)

# 定义内核大小，根据线条的粗细调整
kernel_size = (3, 3)  # 可以根据你的线条粗细调整
kernel = np.ones(kernel_size, np.uint8)

# 进行形态学膨胀操作
dilated = cv2.dilate(image, kernel, iterations=1)

# 进行形态学闭操作
closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

# 保存结果
cv2.imwrite('connected_lines.png', closed)

# 显示结果
cv2.imshow('image', image)
cv2.imshow('connected', closed)
cv2.waitKey(0)
cv2.destroyAllWindows()
