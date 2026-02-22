import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

def local_gaussian_deformations(X, Y, num_gauss_deform):
    shape = X.shape
    u_gx = np.zeros(shape)
    u_gy = np.zeros(shape)
    du_gxdx = np.zeros(shape)
    du_gydy = np.zeros(shape)
    du_gxdy = np.zeros(shape)
    du_gydx = np.zeros(shape)
    for _ in range(num_gauss_deform):
        # randomize parameters
        A_x = random.uniform(0.003, 0.6)
        A_y = random.uniform(0.003, 0.6)
        sigma_x0 = random.uniform(0.06, 0.5)
        sigma_y0 = random.uniform(0.06, 0.5)
        sigma_x1 = random.uniform(0.06, 0.5)
        sigma_y1 = random.uniform(0.06, 0.5)
        x0 = random.randint(0, 255)
        y0 = random.randint(0, 255)
        x1 = random.randint(0, 255)
        y1 = random.randint(0, 255)

        # the exponent
        x_expn = -0.5 * ((X - x0) / sigma_x0)**2 - 0.5 * ((Y - y0) / sigma_y0)**2
        y_expn = -0.5 * ((Y - y1) / sigma_y1)**2 - 0.5 * ((X - x1) / sigma_x1)**2

        # the gaussian deformation
        gauss_x = A_x * np.exp( x_expn )
        gauss_y = A_y * np.exp( y_expn )

        # the partial derivatives of the gaussian deformations
        dgauss_xdx = - gauss_x * (X - x0) / sigma_x0**2
        dgauss_ydy = - gauss_y * (Y - y1) / sigma_y1**2

        dgauss_xdy = - gauss_x * (Y - y0) / sigma_y0**2
        dguass_ydx = - gauss_y * (X - x1) / sigma_x1**2

        # many gaussian deformations
        u_gx += gauss_x
        u_gy += gauss_y

        # many partial derivatives of gaussian deformations for strain fields
        du_gxdx += dgauss_xdx
        du_gydy += dgauss_ydy
        du_gxdy += dgauss_xdy
        du_gydx += dguass_ydx

    return u_gx, u_gy, du_gxdx, du_gydy, du_gxdy, du_gydx


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
            rx = random.randint(min_radius, max_radius)
            ry = random.randint(min_radius, max_radius)
            angle = random.uniform(0, 360)
            intensity = random.randint(0, 150) if grayscale_variation else 0  # Random grayscale intensity. suggested 0.8 - 1.0 (1.0 is completely black)
            if random.random() < 0.5:
                cv2.circle(image, (x, y), radius, (intensity,), -1)
            else:
                cv2.ellipse(image, (x, y), (rx, ry), angle, 0, 360, intensity, -1)
        #print("speckled image data type", image.dtype)
        return image
    

    
    def apply_deformation(self, image, t_x, t_y, k_x, k_y, theta, gamma_x, gamma_y, num_gauss_deform):
        """
        Applies a deformation field to the speckle pattern.
        u, v are the forward displacement
        Exx, Eyy, Exy are the strain fields
        """

        X, Y = np.meshgrid(np.arange(self.width), np.arange(self.height)) # X: X-coordinates of pixels, Y: Y-coordinates of pixels
        # print("X",X)
        # print("Y",Y)

        # Shift window with randomization
        X += random.randint(0, 256) * np.ones_like(X)
        Y += random.randint(0, 256) * np.ones_like(Y)

        # Flip X, Y with randomization
        flip_type_X = random.choice(["none", "horizontal"])
        flip_type_Y = random.choice(["none", "vertical"])
        if flip_type_X == "horizontal":
            X = np.fliplr( X )
            #print("X fliped",X)
        if flip_type_Y == "vertical":
            Y = np.flipud( Y )
            #print("Y fliped",Y)

        # [X,Y]^T vector
        XY = np.stack( (X.ravel(),Y.ravel()), axis=0 )
        # print("XY shape", XY.shape)

        # Rigid body rotation
        R = np.array([[np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]])
        
        # Uniform strech and shear
        D = np.array([[k_x - 1, gamma_x],
                      [gamma_y, k_y - 1]])
        
        # Rigid body translation
        T = np.array([[t_x],
                      [t_y]])
        
        # 2D Gaussian Deformations 
        u_gx, u_gy, du_gxdx, du_gydy, du_gxdy, du_gydx = local_gaussian_deformations(X, Y, num_gauss_deform)
        # print("u_gx shape", u_gx.shape)
        # print("u_gy shape", u_gy.shape)

        u_g = np.array([u_gx.ravel(),
                      u_gy.ravel()]) 
        # print("u_g shape", u_g.shape)

        # Compute displacement fields
        uv = R @ ( D @ XY + u_g ) + T     
        u, v = uv
        u = u.reshape(self.width, self.height)
        v = v.reshape(self.width, self.height)

        # Compute strain fields
        Exx = k_x * np.ones_like(X) + du_gxdx
        Eyy = k_y * np.ones_like(X) + du_gydy
        Exy = 0.5 * (gamma_x * np.ones_like(X) + gamma_y * np.ones_like(X) + du_gxdy + du_gydx)
       
        # Compute new grid coordinates
        X_new = X + u
        Y_new = Y + v
        
        # Apply the transformation using interpolation
        deformed_image = griddata((X_new.ravel(), Y_new.ravel()), image.ravel(), (X, Y), method='linear', fill_value=np.nan) # deformed_image.dtype = float64
               
        return deformed_image, u, v, Exx, Eyy, Exy
    

    def fill_gaps(self, deformed_image, dots_density, min_radius, max_radius, grayscale_variation = False): 
        mask_nan = np.isnan(deformed_image)
        #print("mask_nan", mask_nan)

        coords_to_fill = np.argwhere(mask_nan)
        #print("coords_to_fill", coords_to_fill)

        # Base white image to draw speckles on
        speckle_patch = np.full((self.height, self.width), 255, dtype=np.uint8)

        # Calculate how many speckles to draw
        num_dots = int(len(coords_to_fill) * dots_density)

        # Draw random speckles (black or grayscale dots) on the gaps
        for _ in range(num_dots):
            y, x = coords_to_fill[random.randint(0, len(coords_to_fill) - 1)]
            radius = random.randint(min_radius, max_radius)
            intensity = random.randint(0, 150) if grayscale_variation else 0
            cv2.circle(speckle_patch, (x, y), radius, (intensity,), -1)

        # Replace NaNs in deformed_image with corresponding values from speckle_patch
        filled = np.where(mask_nan, speckle_patch, deformed_image)
        #print("Nan in filled", np.argwhere(filled==np.nan))
        
        # Final cleanup and type conversion
        filled = np.clip(filled, 0, 255).astype(np.uint8)

        return filled


    def blur_image(self, image, blur_strength):
        # Optional: Apply Gaussian blur to simulate camera capture
        image = cv2.GaussianBlur(image, (5, 5), sigmaX= blur_strength)  
        #print("blured image data type", image.dtype)
        return image
    
    def add_gaussian_noise(self, image, mean=0, std=5):
        noise = np.random.normal(mean, std, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def add_vignette(self, image, strength=0.5):
        rows, cols = image.shape
        X_resultant_kernel = cv2.getGaussianKernel(cols, cols*strength)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, rows*strength)
        kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = 255 * kernel / np.max(kernel)
        return np.clip(image * mask, 0, 255).astype(np.uint8)
    
    def motion_blur(self, image, kernel_size=5, angle=0):
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), angle, 1.0), (kernel_size, kernel_size))
        kernel /= kernel.sum()
        return cv2.filter2D(image, -1, kernel)
    
    def brightness_contrast_randomization(self, image, alpha, beta):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    



if __name__ == "__main__":
    # Define save data path
    save_data_path = 'generated_data_DeepDIC'
    # Creat data path if not already exists
    os.makedirs(save_data_path, exist_ok=True)

    # Define number of samples to be generated 
    num_samples = 1
    # Define shape of the image
    image_width, image_height = 256, 256

    for idx in range(num_samples):
        #dots_density = 0.025
        #dots_density = 0.015
        #dots_density = 0.065
        dots_density = random.uniform(0.015, 0.065)
        min_radius = 1                  # Minimum speckle radius
        max_radius = 3                  # Maximum speckle radius
        if random.random() < 0.5: # 50% of the image pairs have grayscale variation
            grayscale_variation = True
        else:
            grayscale_variation = False

        blur_strength = random.uniform(0.5, 1)

        mean_add_gaussian_noise = 0
        std_add_gaussian_noise = 5

        mean_extra_gaussian_noise = 0
        std_extra_gaussian_noise = 10

        alpha_brightness = random.uniform(0.8, 1.2)
        beta_brightness = random.uniform(-20, 20)

        # Deformation parameters: t_x, t_y, k_x, k_y, theta, gamma_x, gamma_y
        deformn_params = {
            't_x' : random.uniform(0, 4), # uniform includes low, but excludes high endpoint
            't_y' : random.uniform(0, 4), 

            'k_x' : random.uniform(0.96, 1.04), 
            'k_y' : random.uniform(0.96, 1.04), 

            'theta' : random.uniform(-0.01, 0.01), 

            'gamma_x' : random.uniform(-0.03, 0.03), 
            'gamma_y' : random.uniform(-0.03, 0.03), 
            
            'num_gauss_deform' : random.randint(0,2) # randint include both endpoints
        }



        data_generator = SpeckleDataGenerator(image_width, image_height)

        reference_image = data_generator.generate_speckle_pattern(dots_density, min_radius, max_radius, grayscale_variation)

        deformed_image, u, v, Exx, Eyy, Exy = data_generator.apply_deformation(reference_image, **deformn_params)

        filled_deformed_image = data_generator.fill_gaps(deformed_image, dots_density, min_radius, max_radius, grayscale_variation)

        
        # Optical Blur / Effects of camera lens 
        if random.random() < 0.8: # 80% of the image pairs have lens blur
            reference_image = data_generator.blur_image(reference_image, blur_strength)
            filled_deformed_image = data_generator.blur_image(filled_deformed_image, blur_strength)

        # Add Camera Gaussian Noise
        if random.random() < 0.7: # 70% have camera gaussian noise
            reference_image = data_generator.add_gaussian_noise(reference_image, mean_add_gaussian_noise, std_add_gaussian_noise)
            filled_deformed_image = data_generator.add_gaussian_noise(filled_deformed_image, mean_add_gaussian_noise, std_add_gaussian_noise)
        elif random.random() > 1 - 0.05: # 5% have extra noise
            reference_image = data_generator.add_gaussian_noise(reference_image, mean_extra_gaussian_noise, std_extra_gaussian_noise)
            filled_deformed_image = data_generator.add_gaussian_noise(filled_deformed_image, mean_extra_gaussian_noise, std_extra_gaussian_noise)


        # Random Contrast
        if random.random() < 0.3: # 30% have random contrast
            reference_image = data_generator.brightness_contrast_randomization(reference_image, alpha=alpha_brightness, beta=beta_brightness)
            filled_deformed_image = data_generator.brightness_contrast_randomization(filled_deformed_image, alpha=alpha_brightness, beta=beta_brightness)
        
        # Save images
        cv2.imwrite(f"{save_data_path}/reference_image_{idx}.png", reference_image)
        cv2.imwrite(f"{save_data_path}/deformed_image_{idx}.png", filled_deformed_image)

        # Save displacement fields u, v
        np.save(f"{save_data_path}/displacement_u_{idx}.npy", u)
        np.save(f"{save_data_path}/displacement_v_{idx}.npy", v)

        # Save strain fields Exx, Eyy, Exy
        np.save(f"{save_data_path}/strain_Exx_{idx}.npy", Exx)
        np.save(f"{save_data_path}/strain_Eyy_{idx}.npy", Eyy)
        np.save(f"{save_data_path}/strain_Exy_{idx}.npy", Exy)

        print(f"Sample {idx} saved: reference_image, filled_deformed_image, u, v, Exx, Eyy, Exy.")

    
    '''Visualization'''
    # Visualize the last generated sample
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    ax[0].imshow(reference_image, cmap='gray', vmin=0, vmax=255)
    ax[0].set_title('Original Speckle Pattern')
    ax[0].axis('off')
    
    ax[1].imshow(deformed_image, cmap='gray', vmin=0, vmax=255)
    ax[1].set_title('Deformed Speckle Pattern')
    ax[1].axis('off')

    ax[2].imshow(filled_deformed_image, cmap='gray', vmin=0, vmax=255)
    ax[2].set_title('Filled Deformed Speckle Pattern')
    ax[2].axis('off')
    
    # Overlay the original and deformed speckle patterns using different colors
    overlay = np.zeros((reference_image.shape[0], reference_image.shape[1], 3), dtype=np.uint8)
    overlay[..., 0] = reference_image  # Red channel for original speckle pattern
    overlay[..., 1] = deformed_image  # Green channel for deformed speckle pattern
    
    ax[3].imshow(overlay)
    ax[3].quiver(np.arange(0, image_width, 20), np.arange(0, image_height, 20), u[::20, ::20], -v[::20, ::20], color='blue') 
    ax[3].set_title('Displacement Field with Overlaid Speckle Patterns')
    ax[3].axis('off')
    
    plt.show()
    