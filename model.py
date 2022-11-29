import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from compressai.ops import LowerBound

SEED=3470

class BinarizeMnist(object):
    def __init__(self):
        pass
    def __call__(self,image):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        return torch.bernoulli(image)

class ScaleMnist(object):
    def __init__(self):
        pass
    def __call__(self,image):
        return 255 * image

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


class VAE_TwoLayer_Alt(nn.Module):
    def __init__(self, h_dim: int = 200, z1_dim: int = 100, im_shape: tuple = (1,28,28)):
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
        self.decoder_1 = nn.Sequential(
            nn.Linear(z1_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, 2 * self.c * self.h *self.w)
        )
        self.eps=1e-5
        self.mcs=5000
        self.mom=0.999
        # self.mom=0.0
        self.lb=LowerBound(self.eps)

    def setseed(self, seed):
        if seed!=-1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

    def _encode_1(self, x):
        x = x.reshape(-1, self.c * self.h * self.w)
        h1_enc = self.encoder_1(x)
        return self.encoder_mu_1(h1_enc), torch.exp(self.encoder_sigma_1(h1_enc))

    def _decode_1(self, z1_hat):
        h1_dec = self.decoder_1(z1_hat)
        return h1_dec.reshape(-1, self.c, self.h, self.w)

    def forward(self, x, mode="sgvb2", return_param=False, seed=-1):
        dist_p_z = torch.distributions.normal.Normal(0,1)
        z1_mu, z1_sigma = self._encode_1(x)
        dist_q_z1_con_x = torch.distributions.normal.Normal(z1_mu, z1_sigma)
        z1_hat = dist_q_z1_con_x.rsample()
        log_q_z1_con_x = dist_q_z1_con_x.log_prob(z1_hat)
        x1_mu = self._decode_1(z1_hat)
        res = x - x1_mu
        z2_mu, z2_sigma = self._encode_1(x - res)
        dist_q_z2_con_z1 = torch.distributions.normal.Normal(z2_mu, z2_sigma)
        z2_hat = dist_q_z2_con_z1.rsample()
        log_q_z2_con_z1 = dist_q_z2_con_z1.log_prob(z2_hat)
        x2_mu = self._decode_1(z2_hat)
        dist_x_con_z1z2 = torch.distributions.normal.Normal(x1_mu + x2_mu, 1.0)
        log_p_x_conz1z2 = dist_x_con_z1z2.log_prob(x)
        log_p_z1 = dist_p_z.log_prob(z1_hat)
        log_p_z2 = dist_p_z.log_prob(z2_hat)
        if mode == "sgvb1":
            elbo = torch.sum(log_p_x_conz1z2, dim=(1,2,3)) +\
                   torch.sum(log_p_z1, dim=1) + torch.sum(log_p_z2, dim=1) -\
                   torch.sum(log_q_z1_con_x, dim=1) -\
                   torch.sum(log_q_z2_con_z1, dim=1)
        elif mode == "sgvb2":
            elbo = torch.sum(log_p_x_conz1z2, dim=(1,2,3)) +\
                   0.5 * torch.sum((1 + 2 * torch.log(z2_sigma) - z2_mu ** 2 - z2_sigma ** 2), dim=1) +\
                   0.5 * torch.sum((1 + 2 * torch.log(z1_sigma) - z1_mu ** 2 - z1_sigma ** 2), dim=1)
        else:
            raise NotImplementedError

        if return_param:
            return elbo, z1_mu, z1_sigma, z2_mu, z2_sigma
        else:
            return elbo

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
        self.eps=1e-5
        self.mcs=5000
        self.mom=0.999
        # self.mom=0.0
        self.lb=LowerBound(self.eps)

    def setseed(self, seed):
        if seed!=-1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

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

    def forward(self, x, mode="sgvb2", return_param=False, seed=-1):
        z1_mu, z1_sigma = self._encode_1(x)
        dist_q_z1_con_x = torch.distributions.normal.Normal(z1_mu, z1_sigma)
        self.setseed(seed)
        z1_hat = dist_q_z1_con_x.rsample()
        log_q_z1_con_x = dist_q_z1_con_x.log_prob(z1_hat)
        z2_mu, z2_sigma = self._encode_2(z1_hat)
        dist_q_z2_con_z1 = torch.distributions.normal.Normal(z2_mu, z2_sigma)
        self.setseed(seed)
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
            return elbo, z1_mu, z1_sigma, z2_mu, z2_sigma
        else:
            return elbo

    def elbo_from_param(self, x, z1_mu, z1_sigma, z2_mu, z2_sigma, mode="sgvb2"):
        dist_q_z1_con_x = torch.distributions.normal.Normal(z1_mu, z1_sigma)
        z1_hat = dist_q_z1_con_x.rsample()
        log_q_z1_con_x = dist_q_z1_con_x.log_prob(z1_hat)
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
        return elbo

    def nll_iwae(self, x, k):
        b, _, _, _ = x.shape
        elbos = torch.zeros([b,k])
        for i in range(k):
            elbos[:, i] = self.forward(x, mode="sgvb1")
        weights = F.softmax(elbos, dim=1)
        iwelbo = torch.sum(elbos * weights, dim=1)
        return iwelbo

    def nll_iwae_from_param(self, x, z1_mu, z1_sigma, z2_mu, z2_sigma, k):
        b, _, _, _ = x.shape
        assert(b == 1)
        rb = 500
        assert(k % rb == 0)
        x = torch.repeat_interleave(x, rb, dim=0)
        z1_mu = torch.repeat_interleave(z1_mu, rb, dim=0)
        z1_sigma = torch.repeat_interleave(z1_sigma, rb, dim=0)
        z2_mu = torch.repeat_interleave(z2_mu, rb, dim=0)
        z2_sigma = torch.repeat_interleave(z2_sigma, rb, dim=0)
        elbos = torch.zeros([b,k])
        for i in range(k // rb):
            elbos[:, i*rb:(i+1)*rb] = self.elbo_from_param(x, z1_mu, z1_sigma, z2_mu, z2_sigma, mode="sgvb1")
        weights = F.softmax(elbos, dim=1)
        iwelbo = torch.sum(elbos * weights, dim=1)
        # return iwelbo
        return torch.mean(elbos, dim=1)

    def savi_naive(self, x, iter, lr, mode="sgvb2", seed=-1):
        b, c, h, w = x.shape
        assert (b==1)
        with torch.no_grad():
            elbo_favi, z1_mu, z1_sigma, z2_mu, z2_sigma = self.forward(x, mode="sgvb2", return_param=True, seed=seed)
        z1_mu = z1_mu.detach().clone().requires_grad_(True)
        z1_sigma = z1_sigma.detach().clone().requires_grad_(True)
        z2_mu = z2_mu.detach().clone().requires_grad_(True)
        z2_sigma = z2_sigma.detach().clone().requires_grad_(True)
        optimizer = optim.SGD([z1_mu, z1_sigma, z2_mu, z2_sigma], lr=lr, momentum=self.mom)
        for i in range(iter):
            dist_q_z1_con_x = torch.distributions.normal.Normal(z1_mu, self.lb(z1_sigma))
            self.setseed(seed)
            z1_hat = dist_q_z1_con_x.rsample()
            log_q_z1_con_x = dist_q_z1_con_x.log_prob(z1_hat)
            dist_q_z2_con_z1 = torch.distributions.normal.Normal(z2_mu, self.lb(z2_sigma))
            self.setseed(seed)
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
            loss = - torch.mean(elbo / (c * h * w), dim=0) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return elbo_favi, elbo

    def savi_approx(self, x, iter, lr, mode="sgvb2", seed=-1):
        b, c, h, w = x.shape
        assert (b==1)
        with torch.no_grad():
            elbo_favi, z1_mu, z1_sigma, z2_mu, z2_sigma = self.forward(x, mode="sgvb2", return_param=True, seed=seed)
        z1_mu = z1_mu.detach().clone().requires_grad_(True)
        z1_sigma = z1_sigma.detach().clone().requires_grad_(True)
        cur_params = [z1_mu, z1_sigma]
        optimizer = optim.SGD(cur_params, lr=lr, momentum=self.mom)
        for i in range(iter):
            dist_q_z1_con_x = torch.distributions.normal.Normal(z1_mu, self.lb(z1_sigma))
            self.setseed(seed)
            z1_hat = dist_q_z1_con_x.rsample()
            log_q_z1_con_x = dist_q_z1_con_x.log_prob(z1_hat)
            z2_mu, z2_sigma = self._encode_2(z1_hat)
            dist_q_z2_con_z1 = torch.distributions.normal.Normal(z2_mu, self.lb(z2_sigma))
            self.setseed(seed)
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
            loss = - torch.mean(elbo / (c * h * w), dim=0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        z1_mu = z1_mu.detach().clone().requires_grad_(False)
        z1_sigma = z1_sigma.detach().clone().requires_grad_(False)
        z2_mu = z2_mu.detach().clone().requires_grad_(True)
        z2_sigma = z2_sigma.detach().clone().requires_grad_(True)
        cur_params = [z2_mu, z2_sigma]
        optimizer = optim.SGD(cur_params, lr=lr, momentum=self.mom)
        for i in range(iter):
            dist_q_z1_con_x = torch.distributions.normal.Normal(z1_mu, self.lb(z1_sigma))
            self.setseed(seed)
            z1_hat = dist_q_z1_con_x.rsample()
            log_q_z1_con_x = dist_q_z1_con_x.log_prob(z1_hat)
            dist_q_z2_con_z1 = torch.distributions.normal.Normal(z2_mu, self.lb(z2_sigma))
            self.setseed(seed)
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
            loss = - torch.mean(elbo / (c * h * w), dim=0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return elbo_favi, elbo

    def savi_approx(self, x, iter, lr, mode="sgvb2", seed=-1):
        pass
    def eval_savi(self, x, iter, lr, mode="sgvb2", seed=-1):
        elbo_favi, elbo_naive = self.savi_naive(x,iter,lr,mode,seed)
        _, elbo_appro = self.savi_approx(x,iter,lr,mode,seed)
        print("[savi] elbo favi: {0:.3f} --- naive: {1:.3f} --- approx: {2:.3f}".format(elbo_favi.item(), elbo_naive.item(), elbo_appro.item()))
        return elbo_favi, elbo_naive, elbo_appro