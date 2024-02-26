This work is few experiments on the idea of using sinusoidal activation function for multi-layer perceptron (MLP), as the paper "Implicit Neural Representations with Periodic Activation Functions". The code is adapted from https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/siren.


In this work, I perform following tasks:

1. While comparing sinusoidal and ReLU activation functions, I also consider Fourier approximants of square and triangle waves as activation functions.
2. I am focussing on 2D reconstruction, where the task of the model is to map coordinates to pixel values.
3. For training purpose, the key idea is to approximate discrete operators like Sobel and Laplacian with partial derviative (achievable via autograd). Essentially $\Delta{I}=\frac{\partial I'}{\partial x}$ and $\Delta^2{I}=\frac{\partial^2 I'}{\partial x^2}$ are assumed. Here $I$ is the origin image and $I'$ is the reconstructed image.
4. Compare the reconstruction from two perspectives. First perspective is to change the activation functions among ReLU, Sine, Triangular and Square wave. Second perspective is the minimization loss by minimizing (essentiall) $||I-I'||_2$, $||\Delta{I}-\frac{\partial I'}{\partial x}||_2$, $||\Delta^2{I}-\frac{\partial^2 I'}{\partial x^2}||_2$ and combination all three.
5. Plotting the layer output distribution over the sine, traingular and sqaure wave activation functions.



# Code
Assuming the pytorch, opencv, scipy and matplotlib are installed in an environment with python>=3.9, the attributes of this work (1-4) are implemented and reproduced by
```bash
python main.py --n_epochs 100 --learning_rate 1e-4 --hidden_features 256 --hidden_layers 2 --img_path \path\to\image.png --n_modes 4

```
