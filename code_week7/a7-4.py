import numpy as np
from skimage.transform import warp, AffineTransform
import matplotlib.pyplot as plt

def transform_image(image, s, theta, t):
    # Compute center c
    c = np.array([image.shape[1] / 2.0, image.shape[0] / 2.0])
    
    # Transformation matrices
    Tc_inv = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]])
    S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    R = np.array([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]])
    Tc = np.array([[1, 0, c[0]], [0, 1, c[1]], [0, 0, 1]])
    Tt = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    
    # Combined transformation matrix
    M = Tt @ Tc @ R @ S @ Tc_inv
    inv_M = np.linalg.inv(M)
    affine_transform = AffineTransform(matrix=inv_M)
    
    # Apply warp
    return warp(image, affine_transform, order=0, preserve_range=True)

image = np.zeros((100, 100))
image[40:60, 40:60] = 1  # Week6 1.4

s = 2 
theta = np.pi / 10 
t = (10.4, 15.7) 

transformed_image = transform_image(image, s, theta, t)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original Image")
ax[1].imshow(transformed_image, cmap='gray')
ax[1].set_title("Transformed Image")
plt.show()

import numpy as np
import pandas as pd

def procrustes_align(source, target):
    centroid_s = np.mean(source, axis=0)
    centroid_t = np.mean(target, axis=0)
    s_centered = source - centroid_s
    t_centered = target - centroid_t

    # Scaling
    scale = np.linalg.norm(t_centered) / np.linalg.norm(s_centered)
    s_scaled = s_centered * scale

    # Rotation
    H = s_scaled.T @ t_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    aligned = (s_scaled @ R) + centroid_t
    return aligned.reshape(-1)

# Load target (first training wing)
train_df = pd.read_csv('BioSCAN dataset Train.csv', header=None)
target = train_df.iloc[0, :-1].values.reshape(-1, 2)

# Align all wings
def align_data(df, target):
    return df.apply(lambda row: procrustes_align(row[:-1].reshape(-1, 2), target), axis=1)

aligned_train = align_data(train_df, target)
aligned_test = align_data(test_df, target)


import matplotlib.pyplot as plt
from skimage import io, filters, color

# Load images
image_nat = io.imread('../TestImages/Week 7/matrikelnumre_nat.png')
# image_art = io.imread('../TestImages/Week 7/matrikelnumre_art.png')

# Convert to grayscale
gray_nat = color.rgb2gray(image_nat)
# gray_art = color.rgb2gray(image_art)

# Compute Otsu threshold
thresh_nat = filters.threshold_otsu(gray_nat)
# thresh_art = filters.threshold_otsu(gray_art)

# Apply threshold to obtain binary masks
binary_nat = gray_nat > thresh_nat
# binary_art = gray_art > thresh_art

# Plot results
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
axes[0, 0].imshow(image_nat)
axes[0, 0].set_title('Original Natural')
# axes[0, 1].imshow(image_art)
# axes[0, 1].set_title('Original Artificial')

axes[1, 0].imshow(gray_nat, cmap='gray')
# axes[1, 1].imshow(gray_art, cmap='gray')
axes[1, 0].set_title('Grayscale Natural')
# axes[1, 1].set_title('Grayscale Artificial')

axes[2, 0].imshow(binary_nat, cmap='gray')
# axes[2, 1].imshow(binary_art, cmap='gray')
axes[2, 0].set_title('Segmented Natural')
# axes[2, 1].set_title('Segmented Artificial')

plt.show()

import numpy as np

# Overlay original binary image and cleaned result
overlay = np.logical_xor(binary_nat, cleaned_nat)  # Highlight changed pixels

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(binary_nat, cmap='gray')
plt.title("Original Binary Image")

plt.subplot(1, 2, 2)
plt.imshow(cleaned_nat, cmap='gray')
plt.imshow(overlay, cmap='hot', alpha=0.6)  # Highlight changes in red
plt.title("Cleaned(Disk=2) with Overlay")

plt.show()


import matplotlib.pyplot as plt
from skimage import morphology

# Define different structuring element sizes
disk_sizes = [1, 2, 3, 4, 5]

fig, axes = plt.subplots(len(disk_sizes), 3, figsize=(15, 10))

for i, size in enumerate(disk_sizes):
    selem = morphology.disk(size)

    # Perform opening and closing
    opened = morphology.opening(binary_nat, selem)
    closed = morphology.closing(binary_nat, selem)

    # Plot original, opened, and closed images
    axes[i, 0].imshow(binary_nat, cmap='gray')
    axes[i, 0].set_title(f'Original (Disk={size})')

    axes[i, 1].imshow(opened, cmap='gray')
    axes[i, 1].set_title(f'Opened (Disk={size})')

    axes[i, 2].imshow(closed, cmap='gray')
    axes[i, 2].set_title(f'Closed (Disk={size})')

plt.tight_layout()
plt.show()


