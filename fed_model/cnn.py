import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import Queue


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)  # 10 output classes for CIFAR-10
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EMAQueue(Queue):
    def __init__(self, size, alpha):
        super().__init__()
        self.shadow = None
        self.size = size
        self.alpha = alpha

    def put(self, item):
        super().put(item)
        if super().qsize() == self.size:
            self._update()

    def _update(self):
        queue_copy = [self.get() for i in range(self.size)]
        if self.shadow is None:
            self.shadow = self.alpha * sum(queue_copy)
        else:
            self.shadow = self.alpha * (self.shadow + sum(queue_copy))

    def volume(self):
        if self.alpha == 1.0:
            return 1.0
        return -1 / (self.alpha - 1)

    def value(self):
        queue_copy = list(self.queue)
        if self.shadow is None:
            return sum(queue_copy)
        else:
            return self.shadow + sum(queue_copy)


class OmegaQueue(Queue):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def put(self, item):
        if super().qsize() == self.size:
            print("Queue Full!")
            super().get()
        super().put(item)

    def value(self):
        queue_copy = list(self.queue)
        return sum(queue_copy)


class CNN_SI(CNN):
    def __init__(self, mode=None, n_clients=5, alpha=0.5, epsilon=0.001):
        super(CNN_SI, self).__init__()
        self.mode = mode
        self.n_clients = n_clients # only required by "mode == queue" and "mode == ema" 
        self.alpha = alpha # only required by "mode == ema"
        self.epsilon = epsilon

    def update_omega(self, W):
        if self.mode == "queue":
            self._update_omega_queue(W)
        elif self.mode == "accumulation":
            self._update_omega_accumulation(W)
        elif self.mode == "ema":
            self._update_omega_ema(W)
        else:
            raise RuntimeError("CNN_SI: update_omega: mode not supported!")

    def surrogate_loss(self):
        if self.mode == "queue":
            return self._surrogate_loss_queue()
        elif self.mode == "accumulation":
            return self._surrogate_loss_accumulation()
        elif self.mode == "ema":
            return self._surrogate_loss_ema()
        else:
            raise RuntimeError("CNN_SI: surrogate_loss: mode not supported!")

    def _update_omega_queue(self, W):
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                p_prev = getattr(self, '{}_SI_prev_task'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega = W[n]/(p_change**2 + self.epsilon)
                try:
                    omega_queue = getattr(self, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega_queue = OmegaQueue(self.n_clients)
                omega_queue.put(omega)

                self.register_buffer('{}_SI_prev_task'.format(n), p_current)
                setattr(self, '{}_SI_omega'.format(n), omega_queue)

    def _update_omega_accumulation(self, W):
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                p_prev = getattr(self, '{}_SI_prev_task'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = W[n]/(p_change**2 + self.epsilon)
                try:
                    omega = getattr(self, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add

                self.register_buffer('{}_SI_prev_task'.format(n), p_current)
                self.register_buffer('{}_SI_omega'.format(n), omega_new)

    def _update_omega_ema(self, W):
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                p_prev = getattr(self, '{}_SI_prev_task'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega = W[n]/(p_change**2 + self.epsilon)
                try:
                    ema_queue = getattr(self, '{}_SI_omega'.format(n))
                except AttributeError:
                    ema_queue = EMAQueue(self.n_clients, self.alpha)
                ema_queue.put(omega)

                self.register_buffer('{}_SI_prev_task'.format(n), p_current)
                setattr(self, '{}_SI_omega'.format(n), ema_queue)

    def _surrogate_loss_queue(self):
        '''Calculate SI's surrogate loss.'''
        try:
            losses = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self, '{}_SI_prev_task'.format(n))
                    omega_queue = getattr(self, '{}_SI_omega'.format(n))
                    omega = omega_queue.value()
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p-prev_values)**2).sum())
            return sum(losses) if len(losses)>0 else torch.tensor(0.).cuda()
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0.).cuda()

    def _surrogate_loss_accumulation(self):
        '''Calculate SI's surrogate loss.'''
        try:
            losses = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self, '{}_SI_prev_task'.format(n))
                    omega = getattr(self, '{}_SI_omega'.format(n))
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p-prev_values)**2).sum())
            return sum(losses) if len(losses)>0 else torch.tensor(0.).cuda()
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0.).cuda()

    def _surrogate_loss_ema(self):
        '''Calculate SI's surrogate loss.'''
        try:
            losses = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self, '{}_SI_prev_task'.format(n))
                    ema_queue = getattr(self, '{}_SI_omega'.format(n))
                    omega = ema_queue.value()
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p-prev_values)**2).sum())
            return sum(losses) if len(losses)>0 else torch.tensor(0.).cuda()
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0.).cuda()




