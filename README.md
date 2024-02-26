This work is few experiments on the idea of using sinusoidal activation function for multi-layer perceptron (MLP), as the paper "Implicit Neural Representations with Periodic Activation Functions". The code is adapted from https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/siren.


In this work, I perform following tasks:

1. While comparing sinusoidal and ReLU activation functions, I also consider Fourier approximants of square and triangle waves.
2. I am focussing on 2D reconstruction, where the task of the model is to map coordinates to pixel values.
3. For training purpose, the key idea is to approximate discrete operators like Sobel and Laplacian with partial derviative (achievable via autograd). Essentially $\Delta{I}=\frac{\partial I'}{\partial x}$ and $\Delta^2{I}=\frac{\partial^2 I'}{\partial x^2}$ are assumed. Here $I$ is the origin image and $I'$ is the reconstructed image.
