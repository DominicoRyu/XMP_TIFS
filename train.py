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

if noise == 'True':
    def create_noise_table(n_bits=8):
        n_bit_combinations = []
        for n in range(n_bits+1):
            all_combinations = list(itertools.combinations(range(n_bits), n))
            n_bit_combinations.append(all_combinations)
        noise_lookup_table = []

        for i in range(len(n_bit_combinations)):
            n_bits_noise = []
            if i == 0:
                binary_list = [0,0,0,0,0,0,0,0]
                n_bits_noise.append(torch.LongTensor(binary_list).to(device)) 
                noise_lookup_table.append(n_bits_noise)
            else:
                for combination in n_bit_combinations[i]:
                    binary_list = [0,0,0,0,0,0,0,0]
                    for index in combination:
                        binary_list[index] = 1
                    n_bits_noise.append(torch.LongTensor(binary_list).to(device)) 
                noise_lookup_table.append(n_bits_noise)

        result_table = []
        for j in range(len(noise_lookup_table)):
            temp_table = torch.stack(noise_lookup_table[j], dim=0)
            result_table.append(temp_table)

        return result_table

    def hex_to_binary(input_data):
        input_data = input_data.to(device) 
        seq_size = input_data.shape[0]

        for i in range(7,-1,-1):
            temp_data = input_data//(2**i)
            input_data -= temp_data*(2**i)

            if i == 7:
                binary_data = temp_data.reshape(-1,1)
            else:
                binary_data = torch.cat((binary_data, temp_data.reshape(-1,1)),dim=1)
        binary_data = binary_data.reshape(8*seq_size)
        return binary_data

    def create_hex_data(input_data, seq_len):
        input_data = input_data.to(device) 
        input_data = input_data.reshape(-1,8)
        augmented_data = input_data * torch.Tensor([128,64,32,16,8,4,2,1]).long().to(device)  # GPU에서 생성
        augmented_data = torch.sum(augmented_data, dim=1)

        if len(augmented_data) != seq_len:
            raise Exception('Augmentation error')

        return augmented_data

    def create_binary_gaussian_noise(noise_lookup_table, distribution, data_size):
        gaussian_noise_sampling = distribution.sample((data_size,)).to(device) 
        num_noise = torch.bincount(gaussian_noise_sampling)
        gaussian_noise_sampling = gaussian_noise_sampling.repeat(8).reshape(-1,data_size)
        gaussian_noise_sampling = gaussian_noise_sampling.T.reshape(-1)

        for n_bits in range(len(num_noise)):
            if num_noise[n_bits] != 0 and n_bits !=0 :
                random_index_list = random.choices(range(len(noise_lookup_table[n_bits])),k=num_noise[n_bits])
                result = noise_lookup_table[n_bits][random_index_list].flatten()

                gaussian_noise_sampling[gaussian_noise_sampling==n_bits] = result

        return gaussian_noise_sampling

    noise_table = create_noise_table(8)  # 이 함수 내에서 GPU로 텐서를 이동시켜야 함
    gaussian_distribution = torch.distributions.Categorical(torch.tensor([0.6826, 0.2718, 0.0428, 0.0026, 0.0002], device=device)) 

    class FFT75_dataset_noise(Dataset):
        def __init__(self, data, label, data_size, prefix, emb_size=emb_size):
            self.x_data = torch.from_numpy(data.copy()).long().to(device) 
            self.y_data = torch.from_numpy(label.copy()).long().to(device) 
            self.data_size = data_size

            self.emb_size = emb_size
            self.prefix = prefix

        def __len__(self):
            return len(self.y_data)

        def __getitem__(self, idx):
            x = self.x_data[idx]
            y = self.y_data[idx]

            if self.prefix == "train":
                augmentation_result = hex_to_binary(x.clone())
                augmentation_result += create_binary_gaussian_noise(noise_table, gaussian_distribution, self.data_size)
                augmentation_result[augmentation_result == 2] = 0
                x = create_hex_data(augmentation_result, self.data_size).long()

            return x, y

else:
    class FFT75_dataset(Dataset):
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

if noise == 'True':
    train_data = FFT75_dataset_noise(train_np_data['x'], train_np_data['y'], seq_size, prefix="train")
    val_data =FFT75_dataset_noise(val_np_data['x'], val_np_data['y'], seq_size, prefix="eval")

else:
    train_data = FFT75_dataset(train_np_data['x'], train_np_data['y'])
    val_data = FFT75_dataset(val_np_data['x'], val_np_data['y'])

# Dataloaders
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)

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

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=first_lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

model = train_model(epochs, model, criterion, optimizer, scheduler, train_loader, val_loader, patience=early_stopping)

wandb.finish()
