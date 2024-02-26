import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import laplace, sobel
from torch.utils.data import Dataset


def siren_init_(weight, is_first=False, omega=1):

    in_features = weight.shape[1]

    with torch.no_grad():
        if is_first:
            bound = 1 / in_features
        else:
            bound = np.sqrt(6 / in_features) / omega

        weight.uniform_(-bound, bound)





class SquarerLayer(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            is_first=False,
            omega=30,  
            custom_init_function_=None,
            n=2, 
    ):
        super().__init__()
        self.omega = omega
        self.n = n  # Store the number of terms
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if custom_init_function_ is None:
            siren_init_(self.linear.weight, is_first=is_first, omega=omega)
        else:
            custom_init_function_(self.linear.weight)


    def forward(self, x):
        #--------Naive implementation----------------
        # square_wave = 0
        # argument = self.omega * self.linear(x)
        # for i in range(1, 2*self.n, 2): 
        #     term = torch.sin(i * argument) / i
        #     square_wave += term
        # square_wave *= (4 / torch.pi)

        # ---- Better optimized--------------------
        argument = self.omega * self.linear(x)
        indices = torch.arange(1, 2 * self.n, 2, device=x.device)
        terms = torch.sin(indices.unsqueeze(0) * argument.unsqueeze(-1)) / indices
        square_wave = terms.sum(dim=-1) * (4 / torch.pi)
        
        return square_wave




class TriangularLayer(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            is_first=False,
            omega=30,  
            custom_init_function_=None,
            n=2, 
    ):
        super().__init__()

        self.omega = omega

        self.n = n  
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if custom_init_function_ is None:
            siren_init_(self.linear.weight, is_first=is_first, omega=omega)
        else:
            custom_init_function_(self.linear.weight)

    def forward(self, x):
        #--------Naive implementation----------------
        # pi_sq = torch.pi**2
        # triangular_wave = 0
        # argument = self.omega * self.linear(x)
        # for i in range(1, 2*self.n, 2):  # Iterate through odd numbers up to 2*n
        #     coefficient = (-1)**((i-1)//2) / i**2
        #     term = coefficient * torch.sin(i * argument)
        #     triangular_wave += term

        # triangular_wave *= (8/pi_sq)
        

        #----------Better optimized----------------
        # Precompute the argument of the sine function
        argument = self.omega * self.linear(x)
        
        # Vectorized computation of the Fourier series for the triangular wave
        indices = torch.arange(1, 2 * self.n, 2, device=x.device)
        coefficients = (-1)**((indices - 1) // 2) / indices**2
        terms = coefficients.unsqueeze(0) * torch.sin(indices.unsqueeze(0) * argument.unsqueeze(-1))
        triangular_wave = terms.sum(dim=-1) * (8 / torch.pi**2)
        
        return triangular_wave



class SineLayer(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            is_first=False,
            omega=30,
            custom_init_function_= None,
    ):
        super().__init__()
        self.omega = omega
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if custom_init_function_ is None:
            siren_init_(self.linear.weight, is_first=is_first, omega=omega)
        else:
            custom_init_function_(self.linear.weight)


    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))



class ImageSiren(nn.Module):
    def __init__(
            self,
            hidden_features,
            hidden_layers=1,
            first_omega=30,
            hidden_omega=30,
            custom_init_function_= None,
            ):
        
        super().__init__()
        in_features = 2
        out_features = 1

        net = []
        net.append(
                SineLayer(
                    in_features,
                    hidden_features,
                    is_first=True,
                    custom_init_function_=custom_init_function_,
                    omega=first_omega,
                    
            )
        )

        for _ in range(hidden_layers):
            net.append(
                    SineLayer(
                        hidden_features,
                        hidden_features,
                        is_first=False,
                        custom_init_function_=custom_init_function_,
                        omega=hidden_omega,
                        
                )
            )

        final_linear = nn.Linear(hidden_features, out_features)

        if custom_init_function_ is None:
            siren_init_(final_linear.weight, is_first=False, omega=hidden_omega)
        else:
            custom_init_function_(final_linear.weight)

        net.append(final_linear)
        self.net = nn.Sequential(*net)


    def forward(self, x):

        return self.net(x)
    


class ImageSirenVariant(nn.Module):
    def __init__(
            self,
            hidden_features,
            hidden_layers=1,
            first_omega=30,
            hidden_omega=30,
            custom_init_function_= None,
            n=2,
            ):
        
        super().__init__()
        in_features = 2
        out_features = 1

        self.n=n

        net = []
        net.append(
                TriangularLayer(
                    in_features,
                    hidden_features,
                    is_first=True,
                    custom_init_function_=custom_init_function_,
                    omega=first_omega,
                    n=self.n,
            )
        )

        for _ in range(hidden_layers):
            net.append(
                    TriangularLayer(
                        hidden_features,
                        hidden_features,
                        is_first=False,
                        custom_init_function_=custom_init_function_,
                        omega=hidden_omega,
                        n=self.n,
                        
                )
            )

        final_linear = nn.Linear(hidden_features, out_features)

        if custom_init_function_ is None:
            siren_init_(final_linear.weight, is_first=False, omega=hidden_omega)
        else:
            custom_init_function_(final_linear.weight)

        net.append(final_linear)
        self.net = nn.Sequential(*net)


    def forward(self, x):

        return self.net(x)
    



class ImageSirenVariant2(nn.Module):
    def __init__(
            self,
            hidden_features,
            hidden_layers=1,
            first_omega=30,
            hidden_omega=30,
            custom_init_function_= None,
            n=2,
            ):
        
        super().__init__()
        in_features = 2
        out_features = 1

        self.n=n

        net = []
        net.append(
                SquarerLayer(
                    in_features,
                    hidden_features,
                    is_first=True,
                    custom_init_function_=custom_init_function_,
                    omega=first_omega,   
                    n=self.n, 
            )
        )

        for _ in range(hidden_layers):
            net.append(
                    SquarerLayer(
                        hidden_features,
                        hidden_features,
                        is_first=False,
                        custom_init_function_=custom_init_function_,
                        omega=hidden_omega,
                        n=self.n,
                        
                )
            )

        final_linear = nn.Linear(hidden_features, out_features)

        if custom_init_function_ is None:
            siren_init_(final_linear.weight, is_first=False, omega=hidden_omega)
        else:
            custom_init_function_(final_linear.weight)

        net.append(final_linear)
        self.net = nn.Sequential(*net)


    def forward(self, x):

        return self.net(x)
    



class ReluMLP(nn.Module):
    def __init__(self, hidden_features, hidden_layers=1):
        super().__init__()

        in_features = 2
        out_features = 1

        layers = [nn.Linear(in_features, hidden_features), nn.ReLU()]

        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_features, out_features))

        self.net = nn.Sequential(*layers)

        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)

    def forward(self, x):
        return self.net(x)





def generate_coordinates(n):

    rows, cols = np.meshgrid(range(n), range(n), indexing="ij")
    coords_abs = np.stack([rows.ravel(), cols.ravel()], axis=-1)

    return coords_abs




class PixelDataset(Dataset):

    def __init__(self, img):
        if not (img.ndim == 2 and img.shape[0] == img.shape[1]):
            raise ValueError("Only 2D square images are supported.")

        self.img = img
        self.size = img.shape[0]
        self.coords_abs = generate_coordinates(self.size)
        self.grad = np.stack([sobel(img, axis=0), sobel(img, axis=1)], axis=-1)
        self.grad_norm = np.linalg.norm(self.grad, axis=-1)
        self.laplace = laplace(img)

    def __len__(self):
        """Determine the number of samples (pixels)."""
        return self.size ** 2

    def __getitem__(self, idx):
        """Get all relevant data for a single coordinate."""
        coords_abs = self.coords_abs[idx]
        r, c = coords_abs

        # nomalization
        coords = 2 * ((coords_abs / self.size) - 0.5)

        return {
            "coords": coords,
            "coords_abs": coords_abs,
            "intensity": self.img[r, c],
            "grad_norm": self.grad_norm[r, c],
            "grad": self.grad[r, c],
            "laplace": self.laplace[r, c],
        }




class GradientUtils:
    @staticmethod
    def gradient(target, coords):
        return torch.autograd.grad(
            target, coords, grad_outputs=torch.ones_like(target), create_graph=True
        )[0]


    @staticmethod
    def divergence(grad, coords):

        div = 0.0
        for i in range(coords.shape[1]):
            div += torch.autograd.grad(
                grad[..., i], coords, torch.ones_like(grad[..., i]), create_graph=True,
            )[0][..., i : i + 1]
        return div


    @staticmethod
    def laplace(target, coords):

        grad = GradientUtils.gradient(target, coords)
        return GradientUtils.divergence(grad, coords)