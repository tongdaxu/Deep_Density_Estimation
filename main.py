import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch}
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(epoch)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path))


class VAE_SingleLayer(nn.Module):
    def __init__(self, h_dim: int = 200, z_dim: int = 50, im_shape: tuple = (1,28,28)):
        super(VAE_SingleLayer, self).__init__()
        self.c, self.h, self.w  = im_shape
        self.encoder = nn.Sequential(
            nn.Linear(self.c * self.h *self.w, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
        )
        self.encoder_mu = nn.Linear(h_dim, z_dim)
        self.encoder_sigma = nn.Sequential(
            nn.Linear(h_dim, z_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, 2 * self.c * self.h *self.w)
        )
        self.eps=1e-4

    def _encode(self, x):
        x = x.reshape(-1, self.c * self.h * self.w)
        h_enc = self.encoder(x)
        return self.encoder_mu(h_enc), torch.exp(self.encoder_sigma(h_enc))

    def _decode(self, z):
        h_dec = self.decoder(z)
        return h_dec.reshape(-1, self.c, self.h, self.w, 2)

    def forward(self, x, mode="sgvb2"):
        z_mu, z_sigma = self._encode(x)
        dist_q_z_con_x = torch.distributions.normal.Normal(z_mu, z_sigma+self.eps)
        z_hat = dist_q_z_con_x.rsample()
        log_q_z_con_x = dist_q_z_con_x.log_prob(z_hat)
        dist_p_z = torch.distributions.normal.Normal(0,1)
        log_p_z = dist_p_z.log_prob(z_hat)
        x_logits = self._decode(z_hat)
        dist_x_con_z = torch.distributions.categorical.Categorical(logits=x_logits)
        log_p_x_con_z = dist_x_con_z.log_prob(x.long())
        if mode == "sgvb1":
            elbo = torch.sum(log_p_x_con_z, dim=(1,2,3)) + torch.sum(log_p_z, dim=1) - torch.sum(log_q_z_con_x, dim=1)
        elif mode == "sgvb2":
            elbo = torch.sum(log_p_x_con_z, dim=(1,2,3)) + 0.5 * torch.sum((1 + 2 * torch.log(z_sigma) - z_mu ** 2 - z_sigma ** 2), dim=1)
        else:
            raise NotImplementedError
        return elbo

    def nll_iwae(self, x, k):
        b, _, _, _ = x.shape
        elbos = torch.zeros([b,k])
        for i in range(k):
            elbos[:, i] = self.forward(x, mode="sgvb1")
        weights = F.softmax(elbos, dim=1)
        iwelbo = torch.sum(elbos * weights, dim=1)
        return iwelbo


class VAE_TwoLayer(nn.Module):
    def __init__(self, h_dim: int = 200, z1_dim: int = 100, z2_dim: int = 50, im_shape: tuple = (1,28,28)):
        super(VAE_TwoLayer, self).__init__()
        self.c, self.h, self.w  = im_shape
        self.encoder_1 = nn.Sequential(
            nn.Linear(self.c * self.h *self.w, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
        )
        self.encoder_mu_1 = nn.Linear(h_dim, z1_dim)
        self.encoder_sigma_1 = nn.Linear(h_dim, z1_dim)
        self.encoder_2 = nn.Sequential(
            nn.Linear(z1_dim, h_dim // 2),
            nn.Tanh(),
            nn.Linear(h_dim // 2, h_dim // 2),
            nn.Tanh(),
        )
        self.encoder_mu_2 = nn.Linear(h_dim // 2, z2_dim)
        self.encoder_sigma_2 = nn.Linear(h_dim // 2, z2_dim)
        self.decoder_2 = nn.Sequential(
            nn.Linear(z2_dim, h_dim // 2),
            nn.Tanh(),
            nn.Linear(h_dim // 2, h_dim // 2),
            nn.Tanh(),
        )
        self.decoder_mu_2 = nn.Linear(h_dim // 2, z1_dim)
        self.decoder_sigma_2 = nn.Linear(h_dim // 2, z1_dim)
        self.decoder_1 = nn.Sequential(
            nn.Linear(z1_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, 2 * self.c * self.h *self.w)
        )
        self.eps=1e-4

    def _encode_1(self, x):
        x = x.reshape(-1, self.c * self.h * self.w)
        h1_enc = self.encoder_1(x)
        return self.encoder_mu_1(h1_enc), torch.exp(self.encoder_sigma_1(h1_enc))

    def _encode_2(self, z1_hat):
        h2_enc = self.encoder_2(z1_hat)
        return self.encoder_mu_2(h2_enc), torch.exp(self.encoder_sigma_2(h2_enc))

    def _decode_2(self, z2_hat):
        h2_dec = self.decoder_2(z2_hat)
        return self.decoder_mu_2(h2_dec), torch.exp(self.decoder_sigma_2(h2_dec))

    def _decode_1(self, z1_hat):
        h1_dec = self.decoder_1(z1_hat)
        return h1_dec.reshape(-1, self.c, self.h, self.w, 2)

    def forward(self, x, mode="sgvb2", return_param=False):
        z1_mu, z1_sigma = self._encode_1(x)
        dist_q_z1_con_x = torch.distributions.normal.Normal(z1_mu, z1_sigma)
        z1_hat = dist_q_z1_con_x.rsample()
        log_q_z1_con_x = dist_q_z1_con_x.log_prob(z1_hat)
        z2_mu, z2_sigma = self._encode_2(z1_hat)
        dist_q_z2_con_z1 = torch.distributions.normal.Normal(z2_mu, z2_sigma)
        z2_hat = dist_q_z2_con_z1.rsample()
        log_q_z2_con_z1 = dist_q_z2_con_z1.log_prob(z2_hat)
        dist_p_z2 = torch.distributions.normal.Normal(0,1)
        log_p_z2 = dist_p_z2.log_prob(z2_hat)
        p_z1_mu, p_z1_sigma = self._decode_2(z2_hat)
        dist_p_z1_con_z2 = torch.distributions.normal.Normal(p_z1_mu, p_z1_sigma)
        log_p_z2_con_z1 = dist_p_z1_con_z2.log_prob(z1_hat)
        x_logits = self._decode_1(z1_hat)
        dist_x_con_z1 = torch.distributions.categorical.Categorical(logits=x_logits)
        log_p_x_con_z1 = dist_x_con_z1.log_prob(x.long())
        if mode == "sgvb1":
            elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
                   torch.sum(log_p_z2_con_z1, dim=1) + torch.sum(log_p_z2, dim=1) -\
                   torch.sum(log_q_z1_con_x, dim=1) -\
                   torch.sum(log_q_z2_con_z1, dim=1)
        elif mode == "sgvb2":
            elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
                   0.5 * torch.sum((1 + 2 * torch.log(z2_sigma) - z2_mu ** 2 - z2_sigma ** 2), dim=1) +\
                   0.5 * torch.sum((1 + 2 * torch.log(z1_sigma) - 2 * torch.log(p_z1_sigma) - (z1_sigma**2 + (z1_mu - p_z1_mu)**2) / (p_z1_sigma**2)), dim=1)
        else:
            raise NotImplementedError
        if return_param:
            return z1_mu, z1_sigma, z2_mu, z2_sigma
        else: 
            return elbo

    def nll_iwae(self, x, k):
        b, _, _, _ = x.shape
        elbos = torch.zeros([b,k])
        for i in range(k):
            elbos[:, i] = self.forward(x, mode="sgvb1")
        weights = F.softmax(elbos, dim=1)
        iwelbo = torch.sum(elbos * weights, dim=1)
        return iwelbo

    def savi_naive(self, x, iter, lr):
        b, _, _, _ = x.shape
        assert (b==1)
        with torch.no_grad():
            z1_mu, z1_sigma, z2_mu, z2_sigma = self.forward(x, mode="sgvb2", return_param=True)
        z1_mu = z1_mu.detach().clone().requires_grad_(True)
        z1_sigma = z1_sigma.detach().clone().requires_grad_(True)
        z2_mu = z1_mu.detach().clone().requires_grad_(True)
        z2_sigma = z2_sigma.detach().clone().requires_grad_(True)
        optimizer = optim.SGD([z1_mu, z1_sigma, z2_mu, z2_sigma], lr=lr)
        for i in range(iter):
            dist_q_z1_con_x = torch.distributions.normal.Normal(z1_mu, z1_sigma)
            z1_hat = dist_q_z1_con_x.rsample()
            dist_q_z2_con_z1 = torch.distributions.normal.Normal(z2_mu, z2_sigma)
            z2_hat = dist_q_z2_con_z1.rsample()
            # log_q_z2_con_z1 = dist_q_z2_con_z1.log_prob(z2_hat)
            dist_p_z2 = torch.distributions.normal.Normal(0,1)
            # log_p_z2 = dist_p_z2.log_prob(z2_hat)
            p_z1_mu, p_z1_sigma = self._decode_2(z2_hat)
            dist_p_z1_con_z2 = torch.distributions.normal.Normal(p_z1_mu, p_z1_sigma)
            # log_p_z2_con_z1 = dist_p_z1_con_z2.log_prob(z1_hat)
            x_logits = self._decode_1(z1_hat)
            dist_x_con_z1 = torch.distributions.categorical.Categorical(logits=x_logits)
            log_p_x_con_z1 = dist_x_con_z1.log_prob(x.long())
            elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
                   0.5 * torch.sum((1 + 2 * torch.log(z2_sigma) - z2_mu ** 2 - z2_sigma ** 2), dim=1) +\
                   0.5 * torch.sum((1 + 2 * torch.log(z1_sigma) - 2 * torch.log(p_z1_sigma) - (z1_sigma**2 + (z1_mu - p_z1_mu)**2) / (p_z1_sigma**2)), dim=1)
            loss = - torch.mean(elbo / (1 * 28 * 28), dim=0) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pass

class Binarize(object):
    def __init__(self):
        pass
    def __call__(self,image):
        return torch.bernoulli(image)

def main(args):

    model_name = args.model
    checkpoint_dir = Path(model_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    transform_train = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(28, scale=(0.8,1.0)),
        transforms.ToTensor(),
        Binarize()
    ])
    transform_test= transforms.Compose([
        transforms.ToTensor(),
        Binarize()
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "VAE_SingleLayer":
        model = VAE_SingleLayer()
    elif args.model == "VAE_TwoLayer":
        model = VAE_TwoLayer()
    else:
        raise NotImplementedError
    model.to(device)
    critical_epochs = [1,3,9,27,81,243,729,2187]
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-04)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.71968)
    training_dataset = datasets.MNIST("./MNIST", train=True, download=True,
                                        transform=transform_train)
    test_dataset = datasets.MNIST("./MNIST", train=False, download=True,
                                    transform=transform_test)
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=True, drop_last=True,
                                 num_workers=args.num_workers, pin_memory=True)
    num_epochs = args.num_epochs
    N = 1 * 28 * 28
    for epoch in range(0, num_epochs + 1):
        ############### training ###############
        model.train()
        average_elbo = average_bpd = 0
        for i, (images, _) in enumerate(training_dataloader, 1):
            images = images.to(device)
            images = torch.round(images)
            elbo = model(images)
            elbo = torch.mean(elbo / N, dim=0)
            loss = -elbo
            bpd = - (elbo / np.log(2))
            average_elbo += -loss.item()
            average_bpd += bpd.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        average_elbo /= i
        average_bpd /= i
        print("[train] epoch:{}, elbo:{:.3f}, bpd:{:.3f}"
              .format(epoch, average_elbo * N, average_bpd))
        ############### testing ###############
        model.eval()
        average_elbo = average_bpd = 0
        for i, (images, _) in enumerate(test_dataloader, 1):
            images = images.to(device)
            with torch.no_grad():
                elbo = model(images)
                elbo = torch.mean(elbo / N, dim=0)
                bpd = - (elbo / np.log(2))
                average_elbo += elbo.item()
                average_bpd += bpd.item()
        average_elbo /= i
        average_bpd /= i
        print("[test] epoch:{}, elbo:{:.3f}, bpd:{:.3f}"
            .format(epoch, average_elbo * N, average_bpd))
        ########### critical points ############
        if epoch in critical_epochs:
            scheduler.step()
            model.eval()
            average_elbo = average_bpd = 0
            for i, (images, _) in enumerate(test_dataloader, 1):
                images = images.to(device)
                with torch.no_grad():
                    elbo = model.nll_iwae(images, 500)
                    elbo = torch.mean(elbo / N, dim=0)
                    bpd = - (elbo / np.log(2))
                    average_elbo += elbo.item()
                    average_bpd += bpd.item()
            average_elbo /= i
            average_bpd /= i
            save_checkpoint(model, optimizer, epoch, checkpoint_dir)
            print("[iwtest] epoch:{}, elbo:{:.3f}, bpd:{:.3f}"
                .format(epoch, average_elbo * N, average_bpd))
    model.eval()
    average_elbo = average_bpd = 0
    for i, (images, _) in enumerate(test_dataloader, 1):
        images = images.to(device)
        with torch.no_grad():
            elbo = model.nll_iwae(images, 5000)
            elbo = torch.mean(elbo / N, dim=0)
            bpd = - (elbo / np.log(2))
            average_elbo += elbo.item()
            average_bpd += bpd.item()
    average_elbo /= i
    average_bpd /= i
    save_checkpoint(model, optimizer, epoch, checkpoint_dir)
    print("[end test] elbo:{:.3f}, bpd:{:.3f}"
        .format(epoch, average_elbo * N, average_bpd))


if __name__ == "__main__":

    SEED=3470
    random.seed(SEED)
    np.random.seed(SEED) 
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic=True
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size.")
    parser.add_argument("--batch-size-test", type=int, default=100, help="Batch size.")
    parser.add_argument("--num-epochs", type=int, default=3280, help="Number of training steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--model", type=str, default="VAE_SingleLayer", help="model type.")

    args = parser.parse_args()
    main(args)