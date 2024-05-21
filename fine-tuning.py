import json
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import os
import wandb
import argparse
import math
import random
import time
import itertools
from models.XMP_model import XMP

with open('config.json', 'r') as f:
    config = json.load(f)

wandb.init(project=config["wandb_project"], name = config["wandb_name"])

training_scenario = config["training_scenario"]
data_size = config["data_size"]
checkpoint = config["checkpoint_name"]
gpu_id = config["gpu_id"]
emb_dim = config["value_embedding"]
depth = config["multi_scale_Performer_depth"]
small_cnn_kernel_size = config["small_cnn_kernel_size"]
small_token_dimension = config["small_token_dimension"]
small_token_size = config["small_token_size"]
large_cnn_kernel_size = config["large_cnn_kernel_size"]
large_token_dimension = config["large_token_dimension"]
large_token_size = config["large_token_size"]
epochs = config["total_epochs"]
first_lr = config["first_lr"]
batch_size = config["dataset_batch_size"]
early_stopping = config["early_stopping"]
mode = config["mode"]
noise = config["noise"]

scenario_name = data_size + '_' + str(training_scenario)
basic_path = './' + scenario_name + '/'

if data_size=='512':
    print('=========== data_size = 512 ===========')
    seq_size = 512
elif data_size=='4k':
    print('=========== data_size = 4096 ===========')
    seq_size = 4096
else:
    raise ValueError('=========== Invalid data size! ===========')

num_class_list = [75,11,25,5,2,2]
num_classes = num_class_list[training_scenario-1]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emb_size = 256

class Fifty_dataset(Dataset):
    def __init__(self, data, label):
        self.x_data = torch.from_numpy(data.copy()).long()
        self.y_data = torch.from_numpy(label.copy()).long()

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y

def train_model(epochs, model, criterion, optimizer, scheduler ,train_loader, val_loader, patience=10):
    best_val_loss = float('inf')
    best_accuracy = float(0)
    epochs_without_improvement = 0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_correct = 0
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            pred_out = outputs.argmax(dim=1, keepdim=True)
            running_correct += pred_out.eq(labels.view_as(pred_out)).sum().item()

        running_loss /= len(train_loader.dataset)
        running_accuracy = 100 * running_correct / len(train_loader.dataset)
        epoch_end_time = time.time()
        epoch_duration = (epoch_end_time - epoch_start_time) / 60

        # Evaluate the model on the validation set
        print('Evaluation start')
        val_loss, val_accuracy = evaluate_model(model, criterion, val_loader)

        print(f'Epoch: {epoch + 1}, Training Loss: {running_loss:.4f}, Training Accuracy: {running_accuracy:.2f}%')
        print(f'Epoch: {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Check if the validation loss has improved
        if val_loss < best_val_loss:
            print('Saving the model...')
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the best model

            name_v1 = './best_model/'+ checkpoint + '_v1.pth'
            name_v2 = './best_model/'+ checkpoint + '_v2.pth'

            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,}, name_v1)

            torch.save(model, name_v2)

        else:
            epochs_without_improvement += 1

        if best_accuracy < val_accuracy:
            best_accuracy = val_accuracy

        print(f'Epoch: {epoch + 1}, Best Accuracy: {best_accuracy:.2f}%')
        print('========================================================')

        # Check if early stopping criteria are met
        if epochs_without_improvement >= patience:
            print(f'Early stopping after {epoch + 1} epochs without improvement.')
            break

        wandb.log({
            "Train Loss": running_loss,
            "Train Acc": running_accuracy,
            "Validation Loss": val_loss,
            "Validation Acc": val_accuracy,
            "Epoch Duration": epoch_duration
        })

        scheduler.step(val_loss)
    print('Finished Training')
    return model

def evaluate_model(model, criterion, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100 * correct / len(val_loader.dataset)
    return val_loss, accuracy

# Load datasets
train_path = basic_path + 'train.npz'
val_path = basic_path + 'val.npz'

train_np_data = np.load(train_path)
val_np_data = np.load(val_path)

train_data = Fifty_dataset(train_np_data['x'], train_np_data['y'])
val_data = Fifty_dataset(val_np_data['x'], val_np_data['y'])

# Dataloaders
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)

#Finetuning
if data_size == "512":
    if noise == "True":
        model_weight = torch.load('./NoiseXMP_scen1_512.pth')
    else:  
        model_weight = torch.load('./XMP_scen1_512.pth')
else:
    if noise == "True":
        model_weight = torch.load('./NoiseXMP_scen1_4k.pth')
    else:
        model_weight = torch.load('./XMP_scen1_4k.pth')

model_state_dict = model_weight['model_state_dict']
model_state_dict = {k: v for k, v in model_state_dict.items() if 'mlp_head' not in k}

model = XMP(
    seq_len = seq_size,
    emb_dim = emb_dim,
    num_classes = num_classes,
    depth = depth,
    sm_cnn_kernel_size = small_cnn_kernel_size,
    sm_dim = small_token_dimension,
    sm_token_size = small_token_size,
    lg_cnn_kernel_size = large_cnn_kernel_size,
    lg_dim = large_token_dimension,
    lg_token_size = large_token_size,
    ).to(device)

model.load_state_dict(model_state_dict, strict=False)
nn.init.trunc_normal_(model.sm_mlp_head[1].weight, std=0.01)
nn.init.trunc_normal_(model.lg_mlp_head[1].weight, std=0.01)

if mode == "adapt":
    print("adaptformer")

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if 'adapter' in name or 'image_embedder' in name:
            param.requires_grad = True

    for param in model.sm_mlp_head.parameters():
        param.requires_grad = True
    for param in model.lg_mlp_head.parameters():
        param.requires_grad = True

elif mode == "vpt":
    print("vpt")
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'vpt' in name or 'image_embedder' in name:
            param.requires_grad = True
    for param in model.sm_mlp_head.parameters():
        param.requires_grad = True
    for param in model.lg_mlp_head.parameters():
        param.requires_grad = True

elif mode == "linear":
    print("Linear")
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'image_embedder' in name:
            param.requires_grad = True
    for param in model.sm_mlp_head.parameters():
        param.requires_grad = True
    for param in model.lg_mlp_head.parameters():
        param.requires_grad = True

else:
    print("original")
    for param in model.parameters():
        param.requires_grad = True

def torchmodify(name) :
    a=name.split('.')
    for i,s in enumerate(a) :
        if s.isnumeric() :
            a[i]="_modules['"+s+"']"
    return '.'.join(a)

for name, module in model.named_modules() :
    if isinstance(module,nn.GELU) :
        exec('model.'+torchmodify(name)+'=nn.GELU()')

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=first_lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

model = train_model(epochs, model, criterion, optimizer, scheduler, train_loader, val_loader, patience=early_stopping)

wandb.finish()
