import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import os

class SpeckleDataGenerator():
    def __init__(self, image_width, image_height):
        self.width = image_width
        self.height = image_height
    
    def generate_speckle_pattern(self, dots_density, min_radius, max_radius, grayscale_variation = False):
        # Calculate speckle dots number based on density
        num_dots = int(self.width * self.height * dots_density)                       
        
        # Create a white background image (grayscale)
        image = np.full((self.height, self.width), 255, dtype=np.uint8)

        # Draw random speckles (black or grayscale dots) on the image
        for _ in range(num_dots):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            radius = random.randint(min_radius, max_radius)
            intensity = random.randint(0, 150) if grayscale_variation else 0  # Random grayscale intensity
            cv2.circle(image, (x, y), radius, (intensity,), -1)

        return image
    
    def blur_image(self, image, blur_strength):
        # Optional: Apply Gaussian blur to simulate camera capture
        image = cv2.GaussianBlur(image, (5, 5), sigmaX= blur_strength)  ### !!!!

        return image
    
    
    def apply_deformation(self, image, alpha, beta):
        """
        Applies a deformation field to the speckle pattern.
        u, v are the backward displacement
        """

        X, Y = np.meshgrid(np.arange(self.width), np.arange(self.height)) # X: X-coordinates of pixels, Y: Y-coordinates of pixels

        # Define displacement fields
        u = alpha * X  # Horizontal displacement (stretching)
        v = beta * Y    # Vertical displacement (shear)
        
        # Compute new grid coordinates
        X_new = X + u
        Y_new = Y + v
        
        # Apply the transformation using interpolation
        deformed_image = map_coordinates(image, [Y_new.ravel(), X_new.ravel()], order=1).reshape(image.shape)
        
        return deformed_image, u, v
        


if __name__ == "__main__":
    # Define save data path
    save_data_path = 'generated_data'
    # Creat data path if not already exists
    os.makedirs(save_data_path, exist_ok=True)

    # Define number of samples to be generated 
    num_samples = 16
    # Define shape of the image
    image_width, image_height = 512, 512

    for idx in range(num_samples):
        dots_density = 0.025
        min_radius = 1                  # Minimum speckle radius
        max_radius = 3                  # Maximum speckle radius

        alpha =  0.1 * (-1 + 2 *random.random())
        beta = 0.1 * (-1 + 2 *random.random())

        data_generator = SpeckleDataGenerator(image_width, image_height)

        reference_image = data_generator.generate_speckle_pattern(dots_density, min_radius, max_radius, grayscale_variation = True)

        deformed_image, u, v = data_generator.apply_deformation(reference_image, alpha, beta)

        # Save images
        cv2.imwrite(f"{save_data_path}/reference_image_{idx}.png", reference_image)
        cv2.imwrite(f"{save_data_path}/deformed_image_{idx}.png", deformed_image)

        # Save displacement fields u, v
        np.save(f"{save_data_path}/displacement_u_{idx}.npy", u)
        np.save(f"{save_data_path}/displacement_v_{idx}.npy", v)

        print(f"Sample {idx} saved: reference_image, deformed_deformed, u, v.")

    
    '''Visualization'''
    # Visualize the last generated sample
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(reference_image, cmap='gray', vmin=0, vmax=255)
    ax[0].set_title('Original Speckle Pattern')
    ax[0].axis('off')
    
    ax[1].imshow(deformed_image, cmap='gray', vmin=0, vmax=255)
    ax[1].set_title('Deformed Speckle Pattern')
    ax[1].axis('off')
    
    # Overlay the original and deformed speckle patterns using different colors
    overlay = np.zeros((reference_image.shape[0], reference_image.shape[1], 3), dtype=np.uint8)
    overlay[..., 0] = reference_image  # Red channel for original speckle pattern
    overlay[..., 1] = deformed_image  # Green channel for deformed speckle pattern
    
    ax[2].imshow(overlay)
    ax[2].quiver(np.arange(0, 512, 20), np.arange(0, 512, 20), u[::20, ::20], -v[::20, ::20], color='blue') 
    ax[2].set_title('Displacement Field with Overlaid Speckle Patterns')
    ax[2].axis('off')
    
    plt.show()