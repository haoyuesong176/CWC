import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import Queue
import timm

from fed_model.vit_small import ViT


class Trans(nn.Module):
    def __init__(self, pretrained=False):
        super(Trans, self).__init__()
        if not pretrained:
            self.vit = ViT(
                image_size = 32,
                patch_size = 4,
                num_classes = 10,
                dim = 512,
                depth = 6,
                heads = 8,
                mlp_dim = 512,
                dropout = 0.1,
                emb_dropout = 0.1
            )
        else:
            self.vit = timm.create_model("vit_base_patch16_384", pretrained=True)
            self.vit.head = nn.Linear(self.vit.head.in_features, 10) 
        
    def forward(self, x):
        x = self.vit(x)
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


class Trans_SI(Trans):
    def __init__(self, pretrained=False, mode=None, n_clients=5, alpha=0.5, epsilon=0.001):
        super(Trans_SI, self).__init__(pretrained)
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
            raise RuntimeError("Trans_SI: update_omega: mode not supported!")

    def surrogate_loss(self):
        if self.mode == "queue":
            return self._surrogate_loss_queue()
        elif self.mode == "accumulation":
            return self._surrogate_loss_accumulation()
        elif self.mode == "ema":
            return self._surrogate_loss_ema()
        else:
            raise RuntimeError("Trans_SI: surrogate_loss: mode not supported!")

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




