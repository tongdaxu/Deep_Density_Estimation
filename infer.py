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
from model import Binarize, VAE_SingleLayer, VAE_TwoLayer

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE_TwoLayer()
    model.to(device)
    checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"], strict=False)
    transform_test= transforms.Compose([
        transforms.ToTensor(),
        Binarize()
    ])
    test_dataset = datasets.MNIST("./MNIST", train=False, download=True,
                                    transform=transform_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True,
                                 num_workers=args.num_workers, pin_memory=True)
    N = 1 * 28 * 28
    model.eval()
    average_elbo_favi = average_elbo_naive = average_eblo_appro = 0
    valid_i = 1
    for i, (images, _) in enumerate(test_dataloader, 1):
        images = images.to(device)
        elbo_favi, elbo_naive, elbo_appro = model.eval_savi(images, args.num_iter, args.learning_rate, "sgvb2", 3470)
        if torch.isnan(elbo_favi).any() or torch.isnan(elbo_naive).any() or torch.isnan(elbo_appro).any():
            # ignore nan sample at this time
            continue
        valid_i += 1
        elbo_favi = torch.mean(elbo_favi / N, dim=0)
        elbo_naive = torch.mean(elbo_naive / N, dim=0)
        elbo_appro = torch.mean(elbo_appro / N, dim=0)
        average_elbo_favi += elbo_favi.item()
        average_elbo_naive += elbo_naive.item()
        average_eblo_appro += elbo_appro.item()
        if i == 500:
            break
    average_elbo_favi /= valid_i
    average_elbo_naive /= valid_i
    average_eblo_appro /= valid_i
    print("[test] elbo favi: {:.3f}, elbo naive: {:.3f}, elbo approx: {:.3f}"
        .format(average_elbo_favi * N, average_elbo_naive * N, average_eblo_appro * N))


if __name__ == "__main__":

    SEED=3470
    random.seed(SEED)
    np.random.seed(SEED) 
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic=True
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--num-iter", type=int, default=50, help="Number of training steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--resume", type=str, default="", help="model load path.")
    args = parser.parse_args()
    main(args)