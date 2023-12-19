import os
import yaml

import torch
import numpy as np
from PIL import Image
from utils.Siamese import SiameseNetwork
from utils.load_transformations import load_transformations
from utils.imshow import imshow
from utils.Grad_CAM import convert_to_tensor, Gram_CAM_heatmap

# Get configuration
with open("config.yml", 'r') as stream:
    params = yaml.safe_load(stream)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = torch.load(os.path.join(params['checkpoints_path'], "model.pth"))
model.eval()
# Load transformation
_, tfms = load_transformations(params)

# Load image 1
# ----------------------------------------------------------------------------------------
A = Image.open('./Data/Flowers/Rose/9167147034_0a66ee3616_n.jpg')

image1 = convert_to_tensor(A, tfms, device).unsqueeze(0)
imshow(image=image1,
       mean=params['image']['normalization']['mean'],
       std=params['image']['normalization']['std'],
       figsize=(3, 3),
       figname='Image1.png')

# Load image 2
# ----------------------------------------------------------------------------------------
B = Image.open('./Data/Flowers/Rose/4612830331_2a44957465_n.jpg')

image2 = convert_to_tensor(B, tfms, device).unsqueeze(0)
imshow(image=image2,
       mean=params['image']['normalization']['mean'],
       std=params['image']['normalization']['std'],
       figsize=(3, 3),
       figname='Image2.png')

# Calculate model's prediction
pred = model(image1, image2)
print(f'Disimilarity: {100 * pred.item():.1f}%')
# Backpropagation
pred.backward()
