import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Linear, ReLU, Sequential
from torch.utils.data import DataLoader
import tqdm
import cv2
from auxils import GradientUtils, ImageSiren, PixelDataset, ReluMLP, ImageSirenVariant, ImageSirenVariant2
import argparse
import os


def main(args):

    targets=["intensity", "grad", "laplace", "all"]
    model_names=['siren','mlp_relu', 'siren_variant', 'siren_variant2']

    # Image loading
    img_ = plt.imread(args.img_path)

    # convert to 2D
    if len(img_.shape) == 3:
        # convert to grayscale
        img_=cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

    if len(img_.shape) > 3 or len(img_.shape) < 2:
        raise ValueError("Unsupported image shape")


    # convert into a square image
    #-----------------------------------------------------
    img = np.zeros((max(img_.shape), max(img_.shape)))
    img[: img_.shape[0], : img_.shape[1]] = img_
    img_ = img.copy()
    #-----------------------------------------------------

    downsampling_factor = 4
    img = 2 * (img_ - 0.5)
    img = img[::downsampling_factor, ::downsampling_factor]
    size = img.shape[0]

    dataset = PixelDataset(img)

    # Parameters
    n_epochs = args.n_epochs
    batch_size = int(size ** 2)


    hidden_features = args.hidden_features
    hidden_layers = args.hidden_layers

    for model_name in model_names:

        ref={
            'img': [dataset.img],
            'grad': [dataset.grad_norm],
            'laplace': [dataset.laplace]
            }
        
        predi={
            'img': [],
            'grad': [],
            'laplace': [],
            'name': []
            }

        for target in targets:

            if not os.path.exists(f"visualization with model_{model_name} and loss target_{target}"):
                os.makedirs(f"visualization with model_{model_name} and loss target_{target}")


            # Model creation
            if model_name == "siren":
                model = ImageSiren(
                    hidden_features,
                    hidden_layers=hidden_layers,
                    hidden_omega=30,
                )
                
            elif model_name == "mlp_relu":
                model = ReluMLP(hidden_features, hidden_layers=hidden_layers)

            elif model_name == "siren_variant":
                model = ImageSirenVariant(
                    hidden_features,
                    hidden_layers=hidden_layers,
                    hidden_omega=30,
                    n=args.n_modes,
                )

            elif model_name == "siren_variant2":
                model = ImageSirenVariant2(
                    hidden_features,
                    hidden_layers=hidden_layers,
                    hidden_omega=30,
                    n=args.n_modes,
                )

            else:
                raise ValueError("Unsupported model")

            dataloader = DataLoader(dataset, batch_size=batch_size)
            optim = torch.optim.Adam(lr=args.learning_rate, params=model.parameters())

            # Training loop
            for e in range(n_epochs):
                losses = []
                for d_batch in tqdm.tqdm(dataloader):
                    x_batch = d_batch["coords"].to(torch.float32)
                    x_batch.requires_grad = True

                    y_true_batch = d_batch["intensity"].to(torch.float32)
                    y_true_batch = y_true_batch[:, None]

                    y_pred_batch = model(x_batch)

                    if target == "intensity":
                        loss = ((y_true_batch - y_pred_batch) ** 2).mean()

                    elif target == "grad":
                        y_pred_g_batch = GradientUtils.gradient(y_pred_batch, x_batch)
                        y_true_g_batch = d_batch["grad"].to(torch.float32)
                        loss = ((y_true_g_batch - y_pred_g_batch) ** 2).mean()

                    elif target == "laplace":
                        y_pred_l_batch = GradientUtils.laplace(y_pred_batch, x_batch)
                        y_true_l_batch = d_batch["laplace"].to(torch.float32)[:, None]
                        loss = ((y_true_l_batch - y_pred_l_batch) ** 2).mean()

                    elif target == "all":
                        loss_i= ((y_true_batch - y_pred_batch) ** 2).mean()
                        
                        y_pred_g_batch = GradientUtils.gradient(y_pred_batch, x_batch)
                        y_true_g_batch = d_batch["grad"].to(torch.float32)
                        loss_g = ((y_true_g_batch - y_pred_g_batch) ** 2).mean()

                        y_pred_l_batch = GradientUtils.laplace(y_pred_batch, x_batch)
                        y_true_l_batch = d_batch["laplace"].to(torch.float32)[:, None]
                        loss_l = ((y_true_l_batch - y_pred_l_batch) ** 2).mean()

                        loss = 1e-1*loss_g + 1e-5*loss_l + loss_i

                    else:
                        raise ValueError("Unrecognized target")

                    losses.append(loss.item())


                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    torch.cuda.empty_cache()                

                print(e, np.mean(losses))

                if e == n_epochs-1:
                    pred_img = np.zeros_like(img)
                    pred_img_grad_norm = np.zeros_like(img)
                    pred_img_laplace =  np.zeros_like(img)

                    orig_img = np.zeros_like(img)
                    for d_batch in tqdm.tqdm(dataloader):
                        coords = d_batch["coords"].to(torch.float32)
                        coords.requires_grad = True
                        coords_abs = d_batch["coords_abs"].numpy()

                        pred = model(coords)
                        pred_n = pred.detach().numpy().squeeze()
                        pred_g = (
                            GradientUtils.gradient(pred, coords)
                            .norm(dim=-1)
                            .detach()
                            .numpy()
                            .squeeze()
                        )
                        pred_l = GradientUtils.laplace(pred, coords).detach().numpy().squeeze()

                        pred_img[coords_abs[:, 0], coords_abs[:, 1]] = pred_n
                        pred_img_grad_norm[coords_abs[:, 0], coords_abs[:, 1]] = pred_g
                        pred_img_laplace[coords_abs[:, 0], coords_abs[:, 1]] = pred_l


    
                    predi['img'].append(pred_img)
                    predi['grad'].append(pred_img_grad_norm)
                    predi['laplace'].append(pred_img_laplace)
                    predi['name'].append(target)


                    fig, axs = plt.subplots(3, 2, constrained_layout=True)
                    axs[0, 0].imshow(dataset.img, cmap="gray")
                    axs[0, 1].imshow(pred_img, cmap="gray")

                    axs[1, 0].imshow(dataset.grad_norm, cmap="gray")
                    axs[1, 1].imshow(pred_img_grad_norm, cmap="gray")

                    axs[2, 0].imshow(dataset.laplace, cmap="gray")
                    axs[2, 1].imshow(pred_img_laplace, cmap="gray")

                    for row in axs:
                        for ax in row:
                            ax.set_axis_off()

                    fig.suptitle(f"Iteration: {e+1}/{n_epochs}")
                    axs[0, 0].set_title("Ground truth")
                    axs[0, 1].set_title("Prediction")

                    plt.savefig(f"visualization with model_{model_name} and loss target_{target}/{e+1}epoch.png")

            del optim
            del model
            torch.cuda.empty_cache()



        num_columns= 1 + len(predi['img'])
        
        fig, axs = plt.subplots(len(ref), num_columns, figsize=(num_columns*2, len(ref['img'])*5))
        fig.suptitle(f"Model: {model_name}")

        checker=True
        for i, (key, ref_images) in enumerate(ref.items()):
            axs[i, 0].imshow(ref_images[0], cmap='gray')
            if checker:
                axs[i, 0].set_title(f"Reference\n {key} (kernel)")

            else:
                axs[i, 0].set_title(f"{key} (kernel)")


            axs[i, 0].axis('off')

            for j, pred_img in enumerate(predi[key]):
                axs[i, j+1].imshow(pred_img, cmap='gray')  
                if checker:
                    axs[i, j+1].set_title(f"Target: {predi['name'][j]}\n {key} (autograd)")
                    if j == len(predi[key])-1:
                        checker=False
                else:
                    axs[i, j+1].set_title(f"{key} (autograd)")

                axs[i, j+1].axis('off')

   
        plt.tight_layout()
        plt.savefig(f"Model_{model_name}.png")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Mapping Coordinates to Intensity')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for training')
    parser.add_argument('--hidden_features', type=int, default=256, help='number of hidden features in the model')
    parser.add_argument('--hidden_layers', type=int, default=2, help='number of hidden layers in the model')
    parser.add_argument('--img_path', type=str, default="meme.png", help='path to the image')
    parser.add_argument('--n_modes', type=int, default=4, help='number of modes in the Fourier approximation')

    args = parser.parse_args()

    main(args)