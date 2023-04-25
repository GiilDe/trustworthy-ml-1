import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PGDAttack:
    """
    White-box L_inf PGD attack using the cross-entropy loss
    """

    def __init__(self, model, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps]. The attack optionally 
        performs random initialization and early stopping, depending on the 
        self.rand_init and self.early_stop flags.
        """
        x_original = torch.clone(x)
        if self.rand_init:
            x = x + torch.FloatTensor(x.shape).uniform_(-self.eps, self.eps)
            x = torch.clip(x, 0, 1)
        for _ in range(self.n):
            x.requires_grad = True
            x.grad = None
            outputs = self.model(x)
            if self.early_stop:
                outputs = outputs.argmax(dim=1)
                mask = outputs == y if targeted else outputs != y
                if torch.sum(mask) == len(y):
                    return x
                mask = ~mask

            loss = self.loss_func(outputs, y)
            if targeted:
                loss = -loss
            loss.sum().backward()
            x.requires_grad = False
            x = x + self.alpha * \
                np.sign(
                    (torch.mul(x.grad, mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                     if self.early_stop else x.grad)
                )
            x = torch.clip(x, x_original - self.eps, x_original + self.eps)
            x = torch.clip(x, 0, 1)

        return x


class NESBBoxPGDAttack:
    """
    Query-based black-box L_inf PGD attack using the cross-entropy loss, 
    where gradients are estimated using Natural Evolutionary Strategies 
    (NES).
    """

    def __init__(self, model, eps=8/255., n=50, alpha=1/255., momentum=0.,
                 k=200, sigma=1/255., rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - momentum: a value in [0., 1.) controlling the "weight" of
              historical gradients estimating gradients at each iteration
        - k: the model is queries 2*k times at each iteration via 
              antithetic sampling to approximate the gradients
        - sigma: the std of the Gaussian noise used for querying
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.momentum = momentum
        self.k = k
        self.sigma = sigma
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns:
        1- The adversarially perturbed samples, which lie in the ranges [0, 1] 
        and [x-eps, x+eps].
        2- A vector with dimensionality len(x) containing the number of queries for
        each sample in x.
        """
        def estimate_grad():
            loss = torch.zeros(x.shape)
            for _ in range(self.k):
                noize = torch.normal(
                    mean=0, std=self.sigma, size=x.shape, dtype=x.dtype, device=x.device)
                noize_minus = -noize
                loss += torch.mul(noize, self.loss_func(self.model(x + noize),
                                  y).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                loss += torch.mul(noize_minus, self.loss_func(self.model(x + noize),
                                  y).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))

            if targeted:
                loss = -loss
            return loss/(2*self.k*self.sigma)

        prev_gradient = torch.zeros_like(x)
        x_original = torch.clone(x)
        n_queries = [self.n]*len(x)
        if self.rand_init:
            x = x + torch.FloatTensor(x.shape).uniform_(-self.eps, self.eps)
            x = torch.clip(x, 0, 1)
        for i in range(self.n):
            print(i)
            outputs = self.model(x)
            if self.early_stop:
                outputs = outputs.argmax(dim=1)
                mask = outputs == y if targeted else outputs != y
                n_queries = [min(n_queries_sample, i) if mask_sample else n_queries_sample for n_queries_sample,
                             mask_sample in zip(n_queries, mask)]
                if torch.sum(mask) == len(y):
                    return x
                mask = ~mask

            grad = self.momentum*prev_gradient + \
                (1 - self.momentum)*estimate_grad()
            grad = grad.detach()
            x = x + self.alpha * \
                np.sign(
                    (torch.mul(grad, mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                     if self.early_stop else grad)
                )
            x = torch.clip(x, x_original - self.eps, x_original + self.eps)
            x = torch.clip(x, 0, 1)

            prev_gradient = grad

        return x, torch.tensor(n_queries)


class PGDEnsembleAttack:
    """
    White-box L_inf PGD attack against an ensemble of models using the 
    cross-entropy loss
    """

    def __init__(self, models, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - models (a sequence): an ensemble of models to attack (i.e., the
              attack aims to decrease their expected loss)
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.models = models
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return torch.stack([model(x) for model in self.models], dim=1).mean(dim=1)

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps].
        """
        x_original = torch.clone(x)
        if self.rand_init:
            x = x + torch.FloatTensor(x.shape).uniform_(-self.eps, self.eps)
            x = torch.clip(x, 0, 1)
        for _ in range(self.n):
            x.requires_grad = True
            x.grad = None
            outputs = self.forward(x)
            if self.early_stop:
                outputs = outputs.argmax(dim=1)
                mask = outputs == y if targeted else outputs != y
                if torch.sum(mask) == len(y):
                    return x
                mask = ~mask

            loss = self.loss_func(outputs, y)
            if targeted:
                loss = -loss
            loss.sum().backward()
            x.requires_grad = False
            x = x + self.alpha * \
                np.sign(
                    (torch.mul(x.grad, mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                     if self.early_stop else x.grad)
                )
            x = torch.clip(x, x_original - self.eps, x_original + self.eps)
            x = torch.clip(x, 0, 1)

        return x
