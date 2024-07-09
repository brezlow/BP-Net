import cv2
import numpy as np
import matplotlib.pyplot as plt

# image_path = 'raw_data/000001.png'
image_path = 'groundtruth/000001.png'

image = cv2.imread(image_path)

R, B, G = cv2.split(image)

distance_map = (R.astype(np.uint32) << 16 | G.astype(
    np.uint32) << 8 | B.astype(np.uint32))


# 将 distance_map 展开为一维数组并排序
flattened_distance_map = distance_map.flatten()
sorted_distance_map = np.sort(flattened_distance_map)

# 将排序后的数据重新形状为图像的原始尺寸
sorted_distance_image = sorted_distance_map.reshape(distance_map.shape)

# 可视化原始 distance_map 和排序后的 distance_map
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Distance Map')
plt.imshow(distance_map, cmap='viridis')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title('Sorted Distance Map')
plt.imshow(sorted_distance_image, cmap='viridis')
plt.colorbar()

plt.show()
