import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from snntorch.functional import ce_rate_loss, ce_temporal_loss, ce_count_loss
import copy


bias = True
# NN Architecture
class CSNNet(nn.Module):
    def __init__(self, input_size,num_steps, beta, spike_grad=None, num_outputs=2):
        super().__init__()
        self.num_steps = num_steps
        self.max_pool_size = 2
        self.conv_kernel = 3
        self.conv_stride = 1
        self.conv_groups = 1
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=self.conv_kernel, stride=self.conv_stride, padding=1, bias=bias)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=1.5, learn_threshold=True)


        lin_size = self.calculate_lin_size(input_size)

        self.fc_out = nn.Linear(lin_size * self.conv1.out_channels, num_outputs)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_threshold=True)
    

    def calculate_lin_size(self, input_size):
        x = torch.zeros(1, 1, input_size)
        x = F.max_pool1d(self.conv1(x), kernel_size=self.max_pool_size)
        lin_size = x.shape[2]
        return lin_size

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.reset_mem()
        mem_out = self.lif_out.reset_mem()

        # Record the final layer
        spk_out_rec = []
        mem_out_rec = []

        for _ in range(self.num_steps): #adicionar prints
            cur1 = F.max_pool1d(self.conv1(x), kernel_size=self.max_pool_size)
            spk, mem1 = self.lif1(cur1, mem1) 

            spk = spk.view(spk.size()[0], -1)

            cur_out = self.fc_out(spk)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)

            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_out)

        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)


def train_csnn(net, optimizer,  train_loader, val_loader, train_config, net_config):
    device, num_epochs, num_steps = train_config['device'],  train_config['num_epochs'], train_config['num_steps']
    loss_type, loss_fn, dtype = train_config['loss_type'], train_config['loss_fn'], train_config['dtype']
    val_fn = train_config['val_net']
    loss_hist = []
    val_acc_hist = []
    val_auc_hist = []
    best_net_list = []
    auc_roc = 0
    loss_val = 0
    #print("Epoch:", end='')
    for epoch in range(num_epochs):
        net.train()
        if (epoch + 1) % 10 == 0: print(f"Epoch:{epoch + 1}|auc:{auc_roc}|loss:{loss_val.item()}")

        # Minibatch training loop
        for data, targets in train_loader:
            data = data.to(device, non_blocking=True).unsqueeze(1)

            targets = targets.to(device, non_blocking=True)

            # forward pass
            spk_rec, mem_rec = net(data)

            # Compute loss
            loss_val = compute_loss(loss_type=loss_type, loss_fn=loss_fn, spk_rec=spk_rec, mem_rec=mem_rec,num_steps=num_steps, targets=targets, dtype=dtype, device=device) 

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())
        _, auc_roc = val_fn(net, device, val_loader, train_config)
      
        best_net_list.append(copy.deepcopy(net.state_dict()))

            #val_acc_hist.extend(accuracy)
        val_auc_hist.extend([auc_roc])


    return net, loss_hist, val_acc_hist, val_auc_hist, best_net_list


def val_csnn(net, device, val_loader, train_config):
    eval_batch = iter(val_loader)
    accuracy = 0
    auc_roc = 0
    all_preds = []
    all_targets = []
    batch_size = train_config['batch_size']
    with torch.no_grad():
        net.eval()
        for data, targets in eval_batch:
            data = data.to(device, non_blocking=True).unsqueeze(1)
            data_size = data.shape[0]
            if data.shape[0] < batch_size:
                last_sample = data[-1].unsqueeze(0)
                num_repeat = batch_size - data.shape[0]
                repeated_samples = last_sample.repeat(num_repeat, *[1] * (data.dim() - 1))

                data = torch.cat([data, repeated_samples], dim=0)
            targets = targets.to(device, non_blocking=True)

            spk_rec, mem_rec = net(data)

            spk_rec = spk_rec[:, :data_size]
            mem_rec = mem_rec[:, :data_size]

            _, predicted = spk_rec.sum(dim=0).max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    auc_roc = roc_auc_score(all_targets, all_preds)
    
    return accuracy, auc_roc


def test_csnn(net,  device, test_loader, train_config):
    all_preds = []
    all_targets = []
    batch_size = train_config['batch_size']

    with torch.no_grad():
        net.eval()
        for data, targets in test_loader:
            data = data.to(device, non_blocking=True).unsqueeze(1)
            data_size = data.shape[0]
            if data.shape[0] < batch_size:
                last_sample = data[-1].unsqueeze(0)
                num_repeat = batch_size - data.shape[0]
                repeated_samples = last_sample.repeat(num_repeat, *[1] * (data.dim() - 1))
                data = torch.cat([data, repeated_samples], dim=0)

            targets = targets.to(device, non_blocking=True)
            # forward pass
            test_spk, _ = net(data)
            test_spk = test_spk[:, :data_size]

            # calculate total accuracy
            _, predicted = test_spk.sum(dim=0).max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return all_preds, all_targets
    

def get_loss_fn(loss_type, class_weights=None, pop_coding=False):
    loss_dict = {
        "rate_loss": ce_rate_loss(weight=class_weights),
        "count_loss": ce_count_loss(weight=class_weights, population_code=pop_coding, num_classes=2),
        "temporal_loss": ce_temporal_loss(weight=class_weights),
        "ce_mem": nn.CrossEntropyLoss(weight=class_weights),
        "bce_loss": nn.BCEWithLogitsLoss(weight=class_weights[1])
    }

    return loss_dict[loss_type]


def compute_loss(loss_type, loss_fn, spk_rec, mem_rec, num_steps, targets, dtype, device):
    loss_val = torch.zeros((1), dtype=dtype, device=device)

    if loss_type in ["rate_loss", "count_loss"]:
        loss_val = loss_fn(spk_rec, targets)
    elif loss_type in ["ce_mem", "bce_loss"]:
        for step in range(num_steps):
            loss_val += loss_fn(mem_rec[step], targets) / num_steps
    elif loss_type == "temporal_loss":
        loss_val = loss_fn(spk_rec, targets)

    return loss_val