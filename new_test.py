import os
import yaml
import torch
from PIL import Image
from utils.Siamese import SiameseNetwork
from utils.load_transformations import load_transformations
from utils.imshow import imshow
from utils.Grad_CAM import convert_to_tensor
import matplotlib.pyplot as plt
from torchvision.transforms import Grayscale

# 创建保存结果的文件夹
results_folder = '/mnt/sda1/template_match/new_results/椅子'
os.makedirs(results_folder, exist_ok=True)

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
Template = Image.open('/mnt/sda1/template_match/template_data/test_data/椅子.jpg').convert('RGB')
image1 = convert_to_tensor(Template, tfms, device).unsqueeze(0)
imshow(image=image1,
       mean=params['image']['normalization']['mean'],
       std=params['image']['normalization']['std'],
       figsize=(3, 3),
       figname='Original_image.png')

# Specify the folder containing the images
image_folder_path = '/mnt/sda1/template_match/match_data/new'

# Initialize variables to store the highest similarity and corresponding image path
max_similarity = 0.0
best_image_path = ''

# Iterate over all images in the folder
for filename in os.listdir(image_folder_path):
    if filename.lower().endswith((".png",".jpg")):
        image_path = os.path.join(image_folder_path, filename)
        B = Image.open(image_path)
        # transform = Grayscale(num_output_channels=3)
        # B = transform(B)
        image2 = convert_to_tensor(B, tfms, device).unsqueeze(0)

        # Calculate model's prediction
        pred = model(image1, image2)
        similarity = pred.item()
        print(f'{filename}的相似度为:  {100 * similarity:.1f}%')

        # Update max_similarity and best_image_path if a higher similarity is found
        if similarity > max_similarity:
            max_similarity = similarity
            best_image_path = image_path

        # 显示并保存每一对图像及其相似度
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(Template)
        ax1.set_title('Template')
        ax1.axis('off')

        ax2.imshow(B)
        ax2.set_title(f'{filename}')
        ax2.axis('off')
        plt.suptitle(f'Similarity: {100 * similarity:.1f}%')

        plt.savefig(os.path.join(results_folder, f'{100 * similarity:.1f}_{filename}_result.png'))
        plt.close()

# Display the image with the highest similarity
best_image = Image.open(best_image_path)
best_image_tensor = convert_to_tensor(best_image, tfms, device).unsqueeze(0)

imshow(image=best_image_tensor,
       mean=params['image']['normalization']['mean'],
       std=params['image']['normalization']['std'],
       figsize=(3, 3),
       figname='Similar_images.png')

plt.close()
plt.close()
# Display both images and their similarity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# Display Image 1
ax1.imshow(Template)
ax1.set_title('Template')
ax1.axis('off')

# Display Image 2 with the highest similarity
ax2.imshow(best_image)
ax2.set_title('Most Similar Image')
ax2.axis('off')

# Display the similarity
plt.suptitle(f'Similarity: {100 * max_similarity:.1f}%')
plt.show()

print(f'The most similar image is: {best_image_path} with similarity: {100 * max_similarity:.1f}%')
