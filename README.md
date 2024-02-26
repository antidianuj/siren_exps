This work is few experiments on the idea of using sinusoidal activation function for multi-layer perceptron (MLP), as the paper "Implicit Neural Representations with Periodic Activation Functions". The code is adapted from https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/siren.


In this work, I perform following tasks:

1. While comparing sinusoidal and ReLU activation functions, I also consider Fourier approximants of square and triangle waves as activation functions.
2. I am focussing on 2D reconstruction, where the task of the model is to map coordinates to pixel values.
3. For training purpose, the key idea is to approximate discrete operators like Sobel and Laplacian with partial derviative (achievable via autograd). Essentially $\Delta{I}=\frac{\partial I'}{\partial x}$ and $\Delta^2{I}=\frac{\partial^2 I'}{\partial x^2}$ are assumed. Here $I$ is the origin image and $I'$ is the reconstructed image.
4. Compare the reconstruction from two perspectives. **First perspective** is to change the activation functions among ReLU, Sine, Triangular and Square wave. **Second perspective** is the minimization loss by minimizing (essentiall) $||I-I'||_2$, $||\Delta{I}-\frac{\partial I'}{\partial x}||_2$, $||\Delta^2{I}-\frac{\partial^2 I'}{\partial x^2}||_2$ and combination of all three.
5. Plotting the layer output distribution over the sine, traingular and sqaure wave activation functions.



# Code
Assuming the pytorch, opencv, scipy and matplotlib are installed in an environment with python>=3.9, the attributes of this work (1-4) are implemented and reproduced by
```bash
python main.py --n_epochs 100 --learning_rate 1e-4 --hidden_features 256 --hidden_layers 2 --img_path \path\to\image.png --n_modes 4
# n_epochs: number of training epochs
# learning_rate: learning rate for training
# hidden_features: number of hidden features in the model
# hidden_layers: number of hidden layers in the model
# img_path: path to the image
# n_modes: number of modes in the Fourier approximation
```

The (5) attribute is implemented in the jupyter notebooks.


# Observations
## Image Reconstruction
The columns represents reference images and reconstructed images with variations of change of loss functions (under **Second perspective**). The rows represent the original, gradiant and Laplacian of image in order.
Following plots show the reconstructions with respect to **First perspective**.

### MLP-RELU
![Model_mlp_relu](https://github.com/antidianuj/siren_exps/assets/47445756/4b3b75f9-12b0-454a-81d6-1b76b03be4ad)

### MLP-SIREN
![Model_siren](https://github.com/antidianuj/siren_exps/assets/47445756/f29f47e3-ac08-48f3-974d-3f2a198df81c)

### MLP-SIREN Variation (Triangular Wave Activation)
![Model_siren_variant](https://github.com/antidianuj/siren_exps/assets/47445756/d662a935-3f86-4f45-83af-8d42a7f852ae)

### MLP-SIREN Variation 2 (Square Wave Activation)
![Model_siren_variant2](https://github.com/antidianuj/siren_exps/assets/47445756/3d7f8bdd-12f7-4ea0-98ef-2788207e20f5)


## Activation Function Distribution
In following I plot and compare the distribution of output over layers for random inputs, with weight initialization of MLP same as that of the paper.

I can make deductions as follows. Triangular wave activation function leads to relatively sparse outputs, and would be better at fitting gradients and laplacians, as compared to original image. On the other hand, square wave seems to roughly have similar distribution as that of sine. But the fact that square wave clearly underperforms possibly illuminates on fact that degeneracy in weight distribution (where the outputs are expected to be dense) leads to inferior performance.
![Slide1](https://github.com/antidianuj/siren_exps/assets/47445756/ebf79cc5-f8c1-44bd-b126-2cc9dc436d86)









