import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F


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
        self.conv2 = nn.Conv1d(in_channels=self.conv1.out_channels, out_channels=8, kernel_size=self.conv_kernel, stride=self.conv_stride,groups=self.conv_groups, padding=1, bias=bias)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_threshold=True)

        lin_size = self.calculate_lin_size(input_size)

        self.fc_out = nn.Linear(lin_size * self.conv2.out_channels, num_outputs, bias=bias)
        self.fc_out = nn.Linear(lin_size * self.conv1.out_channels, num_outputs)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_threshold=True)
    

    def calculate_lin_size(self, input_size):
        x = torch.zeros(1, 1, input_size)
        x = F.max_pool1d(self.conv1(x), kernel_size=self.max_pool_size)
        x = F.max_pool1d(self.conv2(x), kernel_size=self.max_pool_size)
        lin_size = x.shape[2]
        return lin_size

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        mem_out = self.lif_out.reset_mem()

        # Record the final layer
        spk_out_rec = []
        mem_out_rec = []

        for _ in range(self.num_steps):
            cur1 = F.max_pool1d(self.conv1(x), kernel_size=self.max_pool_size)
            spk, mem1 = self.lif1(cur1, mem1) 

            cur2 = F.max_pool1d(self.conv2(spk), kernel_size=self.max_pool_size)
            spk, mem2 = self.lif2(cur2, mem2)

            spk = spk.view(spk.size()[0], -1)

            cur_out = self.fc_out(spk)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)

            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_out)

        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)