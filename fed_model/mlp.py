import torch
import torch.nn as nn
from queue import Queue


class MLP(nn.Module):
    def __init__(self, dataset, hs=None):
        super().__init__()
        if dataset == 'mnist' or dataset == 'mnistd':
            input_size = 784
            hidden_size = 256
        elif dataset == 'cifar10' or dataset == 'cifar10d':
            input_size = 3072
            hidden_size = 1024
        elif dataset == 'pmnist':
            input_size = 784
            hidden_size = 2000
        else:
            raise RuntimeError("MLP: dataset not supported!")

        if hs is not None:
            hidden_size = hs

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


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


class MLP_SI(MLP):
    def __init__(self, dataset, mode=None, n_clients=5, alpha=0.5, epsilon=0.001, hs=None):
        super(MLP_SI, self).__init__(dataset, hs=hs)
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
            raise RuntimeError("MLP_SI: update_omega: mode not supported!")

    def surrogate_loss(self):
        if self.mode == "queue":
            return self._surrogate_loss_queue()
        elif self.mode == "accumulation":
            return self._surrogate_loss_accumulation()
        elif self.mode == "ema":
            return self._surrogate_loss_ema()
        else:
            raise RuntimeError("MLP_SI: surrogate_loss: mode not supported!")

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




