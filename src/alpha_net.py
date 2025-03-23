#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np
from typing import Tuple, List

class BoardData(Dataset):
    def __init__(self, dataset: List):
        # print(f"Dataset type: {type(dataset)}")
        # print(f"First item type: {type(dataset[0])}")
        # print(f"First item length: {len(dataset[0])}")
        
        self.X = []
        self.y_p = []
        self.y_v = []
        
        # Process each item in the dataset
        for item in dataset:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                board_state, policy, value = item
            else:
                print(f"Unexpected item structure: {item}")
                continue
                
            self.X.append(board_state)
            self.y_p.append(policy)
            # Make sure value is a scalar, not an array
            if isinstance(value, (list, tuple, np.ndarray)):
                value = value[0] if len(value) > 0 else 0.0
            self.y_v.append(float(value))
        
        # Convert to tensors
        self.X = torch.stack([torch.from_numpy(np.array(x)).float() for x in self.X])
        self.y_p = torch.stack([torch.from_numpy(np.array(p)).float() for p in self.y_p])
        self.y_v = torch.tensor(self.y_v, dtype=torch.float32).view(-1, 1)
        
        print(f"Final shapes - X: {self.X.shape}, y_p: {self.y_p.shape}, y_v: {self.y_v.shape}")
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y_p[idx], self.y_v[idx]

class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_size = 8*8*73
        self.conv1 = nn.Conv2d(22, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s = s.view(-1, 22, 8, 8)
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes: int = 256, planes: int = 256, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
    
class OutBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Value head
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8*8, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(128)
        self.policy_fc = nn.Linear(8*8*128, 8*8*73)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Value head
        v = F.relu(self.value_bn(self.value_conv(s)))
        v = v.view(-1, 8*8)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(s)))
        p = p.view(-1, 8*8*128)
        p = self.policy_fc(p)
        p = self.logsoftmax(p).exp()
        return p, v
    
class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock()
        self.res_blocks = nn.ModuleList([ResBlock() for _ in range(19)])
        self.outblock = OutBlock()
    
    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.conv(s)
        for res_block in self.res_blocks:
            s = res_block(s)
        return self.outblock(s)

class AlphaLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_value: torch.Tensor, value: torch.Tensor, 
                y_policy: torch.Tensor, policy: torch.Tensor) -> torch.Tensor:
        # print(f"y_value shape: {y_value.shape}, value shape: {value.shape}")
        # print(f"y_policy shape: {y_policy.shape}, policy shape: {policy.shape}")
        
        # Ensure y_value is properly shaped
        if y_value.dim() == 1:
            y_value = y_value.unsqueeze(1)  # Convert to [batch_size, 1]
        if value.dim() == 1:
            value = value.unsqueeze(1)      # Convert to [batch_size, 1]
            
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy * 
                                (1e-6 + y_policy.float()).float().log()), 1)
        
        # print(f"value_error shape: {value_error.shape}, policy_error shape: {policy_error.shape}")
        
        # Ensure dimensions match before adding
        batch_size = policy_error.size(0)
        value_error = value_error.view(batch_size)  # Reshape to match policy_error
        
        total_error = (value_error.float() + policy_error).mean()
        return total_error
    
def train(net: nn.Module, dataset: np.ndarray, epoch_start: int = 0, 
          epoch_stop: int = 20, cpu: int = 0) -> None:
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        # Assign GPU based on process ID
        gpu_id = cpu % num_gpus
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Process {cpu} using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print(f"Process {cpu} using CPU")
    
    torch.manual_seed(cpu)
    
    net.to(device)
    net.train()
    criterion = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                             milestones=[100,200,300,400], 
                                             gamma=0.2)
    
    train_set = BoardData(dataset)
    train_loader = DataLoader(train_set, batch_size=30, shuffle=True, 
                            num_workers=4, pin_memory=True)
    
    losses_per_epoch = []
    for epoch in range(epoch_start, epoch_stop):
        total_loss = 0.0
        losses_per_batch = []
        
        for i, data in enumerate(train_loader, 0):
            state, policy, value = [d.to(device) for d in data]
            # print(f"Batch data shapes - state: {state.shape}, policy: {policy.shape}, value: {value.shape}")
            
            optimizer.zero_grad()
            policy_pred, value_pred = net(state)
            # print(f"Model output shapes - policy_pred: {policy_pred.shape}, value_pred: {value_pred.shape}")
            
            loss = criterion(value_pred, value, policy_pred, policy)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()  # Move scheduler step after optimizer step
            
            total_loss += loss.item()
            
            if i % 10 == 9:
                print(f'Process ID: {os.getpid()} [Epoch: {epoch + 1}, '
                      f'{(i + 1)*30}/{len(train_set)} points] '
                      f'total loss per batch: {total_loss/10:.3f}')
                print(f"Policy: {policy[0].argmax().item()} "
                      f"{policy_pred[0].argmax().item()}")
                print(f"Value: {value[0].item()} {value_pred[0,0].item()}")
                losses_per_batch.append(total_loss/10)
                total_loss = 0.0
        
        # Add the average loss for this epoch
        if len(losses_per_batch) > 0:
            losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        else:
            # If no batches completed, use a default value or skip
            print("Warning: No batches completed in this epoch")
            losses_per_epoch.append(0.0)
        
        # Early stopping
        if len(losses_per_epoch) > 100:
            if abs(sum(losses_per_epoch[-4:-1])/3 - 
                  sum(losses_per_epoch[-16:-13])/3) <= 0.01:
                break

    # Plot training loss
    fig = plt.figure()
    ax = fig.add_subplot(222)
    ax.scatter(range(1, len(losses_per_epoch)+1, 1), losses_per_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per batch")
    ax.set_title("Loss vs Epoch")
    plt.savefig(os.path.join("./model_data/", 
                            f"Loss_vs_Epoch_{datetime.datetime.today().strftime('%Y-%m-%d')}.png"))
    print('Finished Training')

