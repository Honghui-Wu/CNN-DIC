# CNN-DIC
## Convolution Neural Network for Digital Image Correlation  
 generate_data_from_ABAQUS.py: Training data generation from ABAQUS simulated cvs files  
 generate_data_DeepDIC: Training data generation using the method from paper DeepDIC  
 generate_data_DeepDIC_realistic.py: To make the generated reference and deformed image pairs more realistic and solve any bug in data shape and data type:
 1. Speckles realism:  
      1.1. The speckles are both circles and ellipses.  
      1.2. The density of speckles is randomized.  
      1.3. 50% of the image pairs have grayscale variation of speckles.  
      1.4. Filled the gaps at the edges of images, to mimic new speckles that moved into the image window when the specimen deformed.  
  
2. Camera realism:  
      2.1. 80% of the image pairs have optical blur, to mimic the effects of the camera lens.  
      2.2. 70% have Gaussian noise, 5% have extra Gaussian noise, to mimic the noise from camera circuits.  
      2.3. 30% have random brightness contrast.

 generate_data_toy.py: Training data generation, a toy  
 cnn_demo.py: CNN encoder-decoder model, a demo
