# Image denoising using autoencoders

## Dataset
CIFAR-10 dataset consisting of:
1. 60,000 color images (50,000 training and 10,000 testing)
2. Each of size 32 x 32 (with 3 RGB channels)
3. 10 classes with 6,000 images per class

## Model/ Algorithm used
Auto-Encoders

## Model Description
1. Convolutional and Transposed Convolutional blocks have been used for the process of encoding and decoding respectively
2. The encoding, as well as the decoding blocks, consists of Convolutional Layer, Batch Normalization and ReLU activation
3. For encoding, 4 convolutional blocks with downsampling (using stride=2) as well as 1 convolutional block without downsampling that encodes an input image of size (32, 32, 3) to size (2, 2, 256)
4. For decoding, 4 deconvolutional blocks with upsampling (using stride=2) and interleaving concatenations (concatenating 1-3, 2-2, ,3-1) as well as 1 final deconvolutional block to decode (or reconstruct) input of size (2, 2, 256) to size (32, 32, 3)

## Noise levels
Noise level 1 – Normal distribution Noise with σ = 0.1 and μ=0.0 added on original image (0 to 1 scale)

Noise level 2 – Normal distribution Noise with σ = 0.3 and μ=0.0 added on original image (0 to 1 scale)

Noise level 3 – Normal distribution Noise with σ = 0.2 and μ=0.0 added on original image (0 to 1 scale)

![alt text](https://github.com/RushiBhatt007/Denoising-Algorithm-on-CIFAR10-dataset/blob/master/Images/noise_levels.png?raw=true)

## Results
The evaluation metric for Denoising used is Mean Squared Error (MSE) and the results for different noise levels are tabulated below:

Noise Level | MSE due to Noise | MSE after Denoising | % MSE Reduction
--- | --- | --- | --- 
Noise Level-1 | 0.0093 | 0.0018 | 80.5455 %
Noise Level-2 | 0.0642 | 0.0053 | 91.6092 %
Noise Level-3 | 0.0628 | 0.0037 | 94.0540 % 

## Visualized Results

Results for noise level 1:

![alt text](https://github.com/RushiBhatt007/Denoising-Algorithm-on-CIFAR10-dataset/blob/master/Images/noise_1.png?raw=true)

Results for noise level 2:

![alt text](https://github.com/RushiBhatt007/Denoising-Algorithm-on-CIFAR10-dataset/blob/master/Images/noise_2.png?raw=true)

Results for noise level 3:

![alt text](https://github.com/RushiBhatt007/Denoising-Algorithm-on-CIFAR10-dataset/blob/master/Images/noise_3.png?raw=true)
