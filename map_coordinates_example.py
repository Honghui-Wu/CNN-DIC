from scipy.ndimage import map_coordinates

# Create a simple 5x5 image
image = np.array([
    [10, 20, 30, 40, 50],
    [60, 70, 80, 90, 100],
    [110, 120, 130, 140, 150],
    [160, 170, 180, 190, 200],
    [210, 220, 230, 240, 250]
], dtype=np.float32)

# Recreate the shifted image with X_new = X + 0.8 and Y_new = Y + 0.8

# Define the new shift amount
X_shift_amount = 0.8
Y_shift_amount = 0

# Define coordinate shifts (move pixels slightly)
X, Y = np.meshgrid(np.arange(5), np.arange(5))
X_new = X + X_shift_amount  
Y_new = Y + Y_shift_amount  

# Apply mapping with bilinear interpolation
shifted_image = map_coordinates(image, [Y_new.ravel(), X_new.ravel()], order=1).reshape(image.shape)

print(shifted_image)

# Create a figure for visualization
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Original image with grid
ax[0].imshow(image, cmap='gray', vmin=0, vmax=255)
ax[0].set_title("Original Image Grid")
ax[0].set_xticks(np.arange(-0.5, 5, 1), minor=True)
ax[0].set_yticks(np.arange(-0.5, 5, 1), minor=True)
ax[0].grid(which="minor", color="red", linestyle='-', linewidth=0.5)

ax[0].scatter([1+X_shift_amount], [1+Y_shift_amount], color='green', label=f"Interpolate from ({1+X_shift_amount},{1+Y_shift_amount})", s=100)
ax[0].legend()

# Shifted image with interpolated point
ax[1].imshow(shifted_image, cmap='gray', vmin=0, vmax=255)
ax[1].set_title("Shifted Image Grid")
ax[1].set_xticks(np.arange(-0.5, 5, 1), minor=True)
ax[1].set_yticks(np.arange(-0.5, 5, 1), minor=True)
ax[1].grid(which="minor", color="red", linestyle='-', linewidth=0.5)
ax[1].scatter([1], [1], color='blue', label="Pixel (1,1)", s=100)
ax[1].legend()

plt.show()