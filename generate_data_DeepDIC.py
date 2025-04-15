import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

def local_gaussian_deformation(X, Y):
    A_x = random.uniform(0.003, 0.6)
    A_y = random.uniform(0.003, 0.6)
    sigma_x0 = random.uniform(0.06, 0.5)
    sigma_y0 = random.uniform(0.06, 0.5)
    sigma_x1 = random.uniform(0.06, 0.5)
    sigma_y1 = random.uniform(0.06, 0.5)
    x0 = random.randint(0, 511)
    y0 = random.randint(0, 511)
    x1 = random.randint(0, 511)
    y1 = random.randint(0, 511)

    x_ex = -0.5 * ((X - x0) / sigma_x0)**2 - 0.5 * ((Y - y0) / sigma_y0)**2
    y_ex = -0.5 * ((Y - y1) / sigma_y1)**2 - 0.5 * ((X - x1) / sigma_x1)**2

    gauss_x = A_x * np.exp( x_ex )
    gauss_y = A_y * np.exp( y_ex )

    dgauss_xdx = - gauss_x * (X - x0) / sigma_x0**2
    dgauss_ydy = - gauss_y * (Y - y1) / sigma_y1**2

    dgauss_xdy = - gauss_x * (Y - y0) / sigma_y0**2
    dguass_ydx = - gauss_y * (X - x1) / sigma_x1**2


    return gauss_x, gauss_y, dgauss_xdx, dgauss_ydy, dgauss_xdy, dguass_ydx

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
            intensity = random.randint(0, 150) if grayscale_variation else 0  # Random grayscale intensity. suggested 0.8 - 1.0 (1.0 is completely black)
            cv2.circle(image, (x, y), radius, (intensity,), -1)

        return image
    
    def blur_image(self, image, blur_strength):
        # Optional: Apply Gaussian blur to simulate camera capture
        image = cv2.GaussianBlur(image, (5, 5), sigmaX= blur_strength)  

        return image
    
    
    def apply_deformation(self, image, t_x, t_y, k_x, k_y, theta, gamma_x, gamma_y, num_gauss_deform):
        """
        Applies a deformation field to the speckle pattern.
        u, v are the forward displacement
        """

        X, Y = np.meshgrid(np.arange(self.width), np.arange(self.height)) # X: X-coordinates of pixels, Y: Y-coordinates of pixels

        # [X,Y]^T vector
        XY = np.stack( (X.ravel(),Y.ravel()), axis=0 )

        # Rigid body rotation
        R = np.array([[np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]])
        
        # Uniform strech and shear
        D = np.array([[k_x - 1, gamma_x],
                      [gamma_y, k_y - 1]])
        
        # Rigid body translation
        T = np.array([[t_x],
                      [t_y]])
        
        # 2D Gaussian Deformation
        u_gx, u_gy = 0, 0
        du_gxdx, du_gydy, du_gxdy, du_gydx = 0, 0, 0, 0
        for _ in range(num_gauss_deform):
            gauss_x, gauss_y, dgauss_xdx, dgauss_ydy, dgauss_xdy, dguass_ydx = local_gaussian_deformation(X, Y)
            u_gx += gauss_x
            u_gy += gauss_y

            # partial derivatives of gaussian deformations for strain fields
            du_gxdx += dgauss_xdx
            du_gydy += dgauss_ydy
            du_gxdy += dgauss_xdy
            du_gydx += dguass_ydx

        # Compute displacement fields
        uv = R @ D @ XY + T     
        u, v = uv
        u = u.reshape(self.width, self.height) + u_gx
        v = v.reshape(self.width, self.height) + u_gy

        # Compute strain fields
        Exx = k_x + du_gxdx
        Eyy = k_y + du_gydy
        Exy = 0.5 * (gamma_x + gamma_y + du_gxdy + du_gydx)
        
        # Compute new grid coordinates
        X_new = X + u
        Y_new = Y + v
        
        # Apply the transformation using interpolation
        deformed_image = griddata((X_new.ravel(), Y_new.ravel()), image.ravel(), (X, Y), method='linear', fill_value=0)
               
        return deformed_image, u, v, Exx, Eyy, Exy
        


if __name__ == "__main__":
    # Define save data path
    save_data_path = 'generated_data_DeepDIC'
    # Creat data path if not already exists
    os.makedirs(save_data_path, exist_ok=True)

    # Define number of samples to be generated 
    num_samples = 1
    # Define shape of the image
    image_width, image_height = 512, 512

    for idx in range(num_samples):
        dots_density = 0.025
        min_radius = 1                  # Minimum speckle radius
        max_radius = 3                  # Maximum speckle radius

        # Deformation parameters: t_x, t_y, k_x, k_y, theta, gamma_x, gamma_y
        t_x = random.uniform(0, 4)
        t_y = random.uniform(0, 4) 

        k_x = random.uniform(0.96, 1.04) 
        k_y = random.uniform(0.96, 1.04) 

        theta = random.uniform(-0.01, 0.01) 

        gamma_x = random.uniform(-0.03, 0.03) 
        gamma_y = random.uniform(-0.03, 0.03) 
        
        num_gauss_deform = random.randint(1,2)


        data_generator = SpeckleDataGenerator(image_width, image_height)

        reference_image = data_generator.generate_speckle_pattern(dots_density, min_radius, max_radius, grayscale_variation = True)

        deformed_image, u, v, Exx, Eyy, Exy = data_generator.apply_deformation(reference_image, t_x, t_y, k_x, k_y, theta, gamma_x, gamma_y, num_gauss_deform)

        # Save images
        cv2.imwrite(f"{save_data_path}/reference_image_{idx}.png", reference_image)
        cv2.imwrite(f"{save_data_path}/deformed_image_{idx}.png", deformed_image)

        # Save displacement fields u, v
        np.save(f"{save_data_path}/displacement_u_{idx}.npy", u)
        np.save(f"{save_data_path}/displacement_v_{idx}.npy", v)

        # Save strain fields Exx, Eyy, Exy
        np.save(f"{save_data_path}/strain_Exx_{idx}.npy", Exx)
        np.save(f"{save_data_path}/strain_Eyy_{idx}.npy", Eyy)
        np.save(f"{save_data_path}/strain_Exy_{idx}.npy", Exy)

        print(f"Sample {idx} saved: reference_image, deformed_image, u, v, Exx, Eyy, Exy.")

    
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
    