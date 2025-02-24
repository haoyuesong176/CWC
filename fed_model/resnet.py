import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from queue import Queue


class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        resnet50 = models.resnet50(pretrained=False)
        
        self.features = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
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


class ResNet_SI(ResNet):
    def __init__(self, mode=None, n_clients=5, alpha=0.5, epsilon=0.001):
        super(ResNet_SI, self).__init__()
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
            raise RuntimeError("ResNet_SI: update_omega: mode not supported!")

    def surrogate_loss(self):
        if self.mode == "queue":
            return self._surrogate_loss_queue()
        elif self.mode == "accumulation":
            return self._surrogate_loss_accumulation()
        elif self.mode == "ema":
            return self._surrogate_loss_ema()
        else:
            raise RuntimeError("ResNet_SI: surrogate_loss: mode not supported!")

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




