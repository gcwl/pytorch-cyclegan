import time
from tqdm.auto import tqdm
from itertools import chain
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from .discriminator import Discriminator
from .generator import Generator
from .utils import rescale, imshow, to_timedict


class CycleGan:
    def __init__(self, config, trainloader_x, trainloader_y, testloader_x, testloader_y, device):
        self.config = config
        self.trainloader_x = trainloader_x
        self.trainloader_y = trainloader_y
        self.testloader_x = testloader_x
        self.testloader_y = testloader_y
        self.device = device
        # D's and G's
        self.Dx = Discriminator(config.channel_size).to(device)
        self.Dy = Discriminator(config.channel_size).to(device)
        self.Gx2y = Generator(config.channel_size, config.num_residuals).to(device)
        self.Gy2x = Generator(config.channel_size, config.num_residuals).to(device)
        # optimizers
        lr = config.learning_rate
        betas = (config.beta1, config.beta2)
        self.Dx_optim = optim.Adam(self.Dx.parameters(), lr=lr, betas=betas)
        self.Dy_optim = optim.Adam(self.Dy.parameters(), lr=lr, betas=betas)
        G_params = chain(self.Gx2y.parameters(), self.Gy2x.parameters())
        self.G_optim = optim.Adam(G_params, lr=lr, betas=betas)
        # loss functions
        self.real_loss = lambda t: F.mse_loss(t, torch.ones_like(t))
        self.fake_loss = lambda t: F.mse_loss(t, torch.zeros_like(t))
        self.cycle_loss = F.l1_loss

    def train_G(self, x, y):
        self.G_optim.zero_grad()
        # generate fakes from reals
        fake_x = self.Gy2x(y)
        fake_y = self.Gx2y(x)
        # reconstructions from fakes
        reconstructed_x = self.Gy2x(fake_y)
        reconstructed_y = self.Gx2y(fake_x)
        # generator loss
        G_loss = (
            self.real_loss(self.Dx(fake_x))
            + self.real_loss(self.Dy(fake_y))
            + self.cycle_loss(x, reconstructed_x) * self.config.cycle_loss_multiplier
            + self.cycle_loss(y, reconstructed_y) * self.config.cycle_loss_multiplier
        )
        G_loss.backward()
        self.G_optim.step()
        return G_loss.item()

    def train_Dx(self, x, y):
        self.Dx_optim.zero_grad()
        # generate fake_x from y
        with torch.no_grad():
            fake_x = self.Gy2x(y)
        # compute real_loss from (real) x, fake_loss from fake_x
        # sum real_loss and fake_loss
        Dx_loss = self.real_loss(self.Dx(x)) + self.fake_loss(self.Dx(fake_x))
        Dx_loss.backward()
        self.Dx_optim.step()
        return Dx_loss.item()

    def train_Dy(self, x, y):
        self.Dy_optim.zero_grad()
        # generate fake_y from (real) x
        with torch.no_grad():
            fake_y = self.Gx2y(x)
        # compute real_loss from (real) y, fake_loss from fake_y
        # sum real_loss and fake_loss
        Dy_loss = self.real_loss(self.Dy(y)) + self.fake_loss(self.Dy(fake_y))
        Dy_loss.backward()
        self.Dy_optim.step()
        return Dy_loss.item()

    def train_one_epoch(self, epoch):
        G_losses, Dx_losses, Dy_losses, batch_sizes = [], [], [], []
        num_iters = min(len(self.trainloader_x), len(self.trainloader_y))
        for (x, _), (y, _) in tqdm(
            zip(self.trainloader_x, self.trainloader_y), total=num_iters, desc=f"Epoch={epoch}"
        ):
            if x.size(0) != y.size(0):
                break
            x = rescale(x.to(self.device))
            y = rescale(y.to(self.device))
            batch_sizes.append(x.size(0))
            G_losses.append(self.train_G(x, y))
            Dx_losses.append(self.train_Dx(x, y))
            Dy_losses.append(self.train_Dy(x, y))
        G_loss = np.average(G_losses, weights=batch_sizes)
        Dx_loss = np.average(Dx_losses, weights=batch_sizes)
        Dy_loss = np.average(Dy_losses, weights=batch_sizes)
        return G_loss, Dx_loss, Dy_loss

    def train(self):
        for epoch in tqdm(range(1, self.config.num_epochs + 1)):
            start_time = time.time()
            G_loss, Dx_loss, Dy_loss = self.train_one_epoch(epoch)
            end_time = time.time()
            self._checkpoint(epoch, start_time, end_time, G_loss, Dx_loss, Dy_loss)

    def _checkpoint(self, *args):
        self._print(*args)
        self._eval(*args)
        self._save(*args)

    def _print(self, *args):
        epoch, start_time, end_time, G_loss, Dx_loss, Dy_loss = args
        if epoch % self.config.print_freq != 0:
            return
        t = to_timedict(end_time - start_time)
        msg = "Epoch: {:03} | G_loss: {:.03f} | Dx_loss: {:.03f} | Dy_loss: {:.03f} | Elapsed: {}m {}s".format(
            epoch, G_loss, Dx_loss, Dy_loss, t["mins"], t["secs"]
        )
        print(msg)

    def _eval(self, *args):
        epoch, start_time, end_time, G_loss, Dx_loss, Dy_loss = args
        if epoch % self.config.eval_freq != 0:
            return
        with torch.no_grad():
            # get fixed_x, if not yet existed
            if not hasattr(self, "fixed_x"):
                x, _ = next(iter(self.testloader_x))
                self.fixed_x = rescale(x.to(self.device))
            # get fixed_y, if not yet existed
            if not hasattr(self, "fixed_y"):
                y, _ = next(iter(self.testloader_y))
                self.fixed_y = rescale(y.to(self.device))
            # generate fakes from (fixed) real images
            fake_y = self.Gx2y(self.fixed_x)
            fake_x = self.Gy2x(self.fixed_y)
            # rescale range for plotting
            im_x = rescale(self.fixed_x, from_range=(-1, 1), to_range=(0, 1))
            im_y = rescale(self.fixed_y, from_range=(-1, 1), to_range=(0, 1))
            im_fake_x = rescale(fake_x, from_range=(-1, 1), to_range=(0, 1))
            im_fake_y = rescale(fake_y, from_range=(-1, 1), to_range=(0, 1))
        # print message
        print(f"Evaluation at epoch={epoch}")
        # plot (X, X-to-Y) at current epoch
        im_x2y = (
            torch.stack([im_x, im_fake_y])  # size = (2, C, H, W)
            .transpose(0, 1)  # swap axes 0 and 1, i.e. size = (C, 2, H, W)
            .reshape(-1, 3, self.config.image_size, self.config.image_size)
        )
        grid_x2y = torchvision.utils.make_grid(im_x2y, nrow=8)
        _, (_, ax) = imshow(grid_x2y, figsize=(20, 20))
        ax.set_title(f"(X, X->Y) at epoch={epoch}")
        # plot (Y, Y-to-X) at current epoch
        im_y2x = (
            torch.stack([im_y, im_fake_x])  # size = (2, C, H, W)
            .transpose(0, 1)  # swap axes 0 and 1, i.e. size = (C, 2, H, W)
            .reshape(-1, 3, self.config.image_size, self.config.image_size)
        )
        grid_y2x = torchvision.utils.make_grid(im_y2x, nrow=8)
        _, (_, ax) = imshow(grid_y2x, figsize=(20, 20))
        ax.set_title(f"(Y, Y->X) at epoch={epoch}")
        # show plots
        plt.show()

    def _save(self, *args):
        epoch, start_time, end_time, G_loss, Dx_loss, Dy_loss = args
        if epoch % self.config.save_freq != 0:
            return
        # TODO
