
import os
import random
import numpy as np
import yaml
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils.load_transformations import load_transformations
from utils.dataset import Dataset
from utils.utils import get_dataset, create_instances, format_time,Caltech256_dataset
from utils.Siamese import SiameseNetwork
from utils.early_stopping import EarlyStopping
from utils.LRScheduler import LRScheduler
from utils.performance_evaluation import performance_evaluation

from torch.utils.tensorboard import SummaryWriter




# Get configuration
with open("config.yml", 'r') as stream:
    params = yaml.safe_load(stream)
if params['cuda']:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建TensorBoard的SummaryWriter
writer = SummaryWriter(log_dir=params['tensorboard_log_dir'])


# Retrieve dataset
df = get_dataset(params) #使用通用数据集
#df = Caltech256_dataset(params) #使用Caltech256数据集
# Split to train/valid/test
df_train, df_test = train_test_split(df, test_size=params['test_size'], random_state=params['seed'])
df_train, df_valid = train_test_split(df_train, test_size=params['valid_size'], random_state=params['seed'])
# Create positive/negative instances
train_dataset = create_instances(df=df_train,
                                 number_of_iterations=params['number_of_iterations'])
valid_dataset = create_instances(df=df_valid,
                                 number_of_iterations=params['number_of_iterations'])
test_dataset = create_instances(df=df_test,
                                number_of_iterations=params['number_of_iterations'])
# Get training/testing image transformations
train_tfms, test_tfms = load_transformations(params)


# Create loaders
train_dl = DataLoader(dataset = Dataset(data=train_dataset, tfms=train_tfms),
                      batch_size  = params['hyperparameters']['batch_size'],
                      shuffle     = True,
                      num_workers = params['hyperparameters']['num_workers'],
                      pin_memory  = True)


valid_dl = DataLoader(dataset = Dataset(data=valid_dataset, tfms=test_tfms),
                      batch_size  = params['hyperparameters']['batch_size'],
                      shuffle     = False,
                      num_workers = params['hyperparameters']['num_workers'],
                      pin_memory  = True)

test_dl = DataLoader(dataset = Dataset(data=test_dataset, tfms=test_tfms),
                      batch_size  = params['hyperparameters']['batch_size'],
                      shuffle     = False,
                      num_workers = params['hyperparameters']['num_workers'],
                      pin_memory  = True)




# Training
# Setup model
model = SiameseNetwork(backbone_model=params['backbone_model']).to(device)


# Setup optimizer
if params['hyperparameters']['optimizer'] == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params['hyperparameters']['learning_rate']))
elif params['hyperparameters']['optimizer'] == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['hyperparameters']['learning_rate']))
elif params['hyperparameters']['optimizer'] == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=float(params['hyperparameters']['learning_rate']))

scheduler = LRScheduler(optimizer=optimizer,
                        patience=params['LRScheduler']['patience'],
                        min_lr=params['LRScheduler']['min_lr'],
                        factor=params['LRScheduler']['factor'],
                        verbose=params['LRScheduler']['verbose'])

# Early stopping
early_stopping = EarlyStopping(patience=params['early_stopping']['patience'],
                               min_delta=params['early_stopping']['min_delta'])

best_AUC = 0.0
history = {'train_loss': [], 'valid_loss': [],
           'train_accuracy': [], 'valid_accuracy': [],
           'train_AUC': [], 'valid_AUC': []}
# 初始化计数器
save_interval = 5
epochs_since_last_save = 0

for epoch in range(params['hyperparameters']['epochs']):

    t0 = time.time()

    # Activate training mode
    model.train()

    # setup loop with TQDM and dataloader
    loop = tqdm(train_dl, leave=True)
    # setup epoch's metrics
    metrics = {'losses': [], 'accuracy': [], 'AUC': []}
    for step, (img1, img2, labels) in enumerate(loop):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        # initialize calculated gradients
        optimizer.zero_grad()
        # Get loss and predictions
        predictions, loss = model(img1, img2, labels)
        # Calculate performance metrics
        accuracy, AUC, _ = performance_evaluation(labels, predictions)
        # Backpropagate errors
        loss.backward()
        # Clip gradient norm
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params['hyperparameters']['max_norm'])
        # update parameters
        optimizer.step()
        # Add loss
        metrics['losses'].append(loss.item())
        metrics['accuracy'].append(accuracy)
        metrics['AUC'].append(AUC)
        # add stuff to progress bar in the end
        loop.set_description(f"Epoch [{epoch + 1}/{params['hyperparameters']['epochs']}]")
        loop.set_postfix(loss=f"{np.mean(metrics['losses']):.3f}",
                         accuracy=f"{np.mean(metrics['accuracy']):.2f}%",
                         AUC=f"{np.mean(metrics['AUC']):.3f}")

        # 记录到TensorBoard
        writer.add_scalar('Train/Loss', np.mean(metrics['losses']), global_step=epoch+1)
        writer.add_scalar('Train/Accuracy', np.mean(metrics['accuracy']), global_step=epoch+1)
        writer.add_scalar('Train/AUC', np.mean(metrics['AUC']), global_step=epoch+1)

    # Calculate test loss/accuracy/AUC
    train_loss = np.mean(metrics['losses'])
    train_accuracy = np.mean(metrics['accuracy'])
    train_AUC = np.mean(metrics['AUC'])


    model.eval()
    ConfusionMatrix = np.array([[0, 0], [0, 0]])  # LIVIERIS
    # setup loop with TQDM and dataloader
    loop = tqdm(valid_dl, leave=True)
    # setup epoch's metrics
    metrics = {'losses': [], 'accuracy': [], 'AUC': []}
    for step, (img1, img2, labels) in enumerate(loop):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        # Get loss & predictions
        predictions, loss = model(img1, img2, labels)
        # Calculate performance metrics
        accuracy, AUC, CM = performance_evaluation(labels, predictions)
        ConfusionMatrix += CM
        # Add loss/accuracy/AUC
        metrics['losses'].append(loss.item())
        metrics['accuracy'].append(accuracy)
        metrics['AUC'].append(AUC)

        # add stuff to progress bar in the end
        loop.set_description(f"Epoch [{epoch + 1}/{params['hyperparameters']['epochs']}]")
        loop.set_postfix(loss=f"{np.mean(metrics['losses']):.3f}",
                         accuracy=f"{np.mean(metrics['accuracy']):.2f}%",
                         AUC=f"{np.mean(metrics['AUC']):.3f}")

        # 记录到TensorBoard
        writer.add_scalar('Validation/Loss', np.mean(metrics['losses']), global_step=epoch+1)
        writer.add_scalar('Validation/Accuracy', np.mean(metrics['accuracy']), global_step=epoch+1)
        writer.add_scalar('Validation/AUC', np.mean(metrics['AUC']), global_step=epoch+1)


    print(ConfusionMatrix)  # LIVIERIS
    # Calculate test loss/MSE
    valid_loss = np.mean(metrics['losses'])
    valid_accuracy = np.mean(metrics['accuracy'])
    valid_AUC = np.mean(metrics['AUC'])

    # Elapsed time per epoch
    elapsed = format_time(time.time() - t0)

    # Store performance
    history['train_loss'].append(train_loss)
    history['valid_loss'].append(valid_loss)
    history['train_accuracy'].append(train_accuracy)
    history['valid_accuracy'].append(valid_accuracy)
    history['train_AUC'].append(train_AUC)
    history['valid_AUC'].append(valid_AUC)

    # Update best model
    if valid_AUC > best_AUC:
        print('[INFO] Model saved')
        if (not os.path.exists(params['checkpoints_path'])):
            os.makedirs(params['checkpoints_path'])
        torch.save(model, os.path.join(params['checkpoints_path'], "model.pth"))
        best_AUC = valid_AUC

        # Learning rate scheduler
    scheduler(valid_AUC)

    # 每经过save_interval个epoch，保存一次模型
    epochs_since_last_save += 1
    if epochs_since_last_save >= save_interval:
        print(f'[INFO] Saving model after {epoch + 1} epochs')
        torch.save(model, os.path.join(params['checkpoints_path'], f"model_epoch_{epoch + 1}.pth"))
        epochs_since_last_save = 0

    # Early Stopping
    #if early_stopping(valid_AUC): break
# 训练结束后关闭TensorBoard的SummaryWriter
writer.close()





