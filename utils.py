import torch
import numpy as np
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_condition_vector(gender,article,color,season):

    cond = torch.zeros(30)

    cond[gender_map[gender]] = 1
    cond[article_map[article]] = 1
    cond[color_map[color]] = 1
    cond[season_map[season]] = 1

    return cond


def generate_images(model,gender,article,color,season,n=4):

    model.eval()

    cond = create_condition_vector(gender,article,color,season)
    cond = cond.unsqueeze(0).repeat(n,1).to(device)

    z = torch.randn(n,128).to(device)

    with torch.no_grad():
        imgs = model.decoder(z,cond)

    imgs = imgs.cpu().numpy()

    return imgs