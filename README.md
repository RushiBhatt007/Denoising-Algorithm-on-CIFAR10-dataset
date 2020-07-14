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
1. Noise level 1 – Normal distribution Noise with σ = 0.1 and μ=0.0 added on original image (0 to 1 scale)
2. Noise level 2 – Normal distribution Noise with σ = 0.3 and μ=0.0 added on original image (0 to 1 scale)
3. Noise level 3 – Normal distribution Noise with σ = 0.2 and μ=0.0 added on original image (0 to 1 scale)
