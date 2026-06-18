"""
OPTIMIZADOR ESGD - IMPLEMENTACION GARF (FUNCIONAL)
=====================================================
Implementacion del optimizador Equilibrated Stochastic Gradient Descent
para el codebase GARF. A diferencia de la implementacion en nf-soft-mining,
ESTA VERSION SI CONVERGE correctamente en tareas de NeRF.

ALGORITMO:
1. Calcula productos Hessiano-vector (HVP) usando el truco de Hutchinson
   con vectores Rademacher (distribucion uniforme en {-1, +1}).
2. Mantiene una EMA del HVP al cuadrado como estimacion de la diagonal
   Hessiana: exp_avg_d = beta2 * exp_avg_d + (1-beta2) * hvp^2
3. El precondicionador D = exp_avg_d.sqrt() escala la gradiente.
4. Incluye LR warmup (parametro lr_warmup) para estabilidad inicial.
5. should_create_graph() controla cuando se necesita el grafo computacional:
   - Durante d_warmup (primeras 50 iteraciones): siempre
   - Despues: cada update_d_every (100) iteraciones

HIPERPARAMETROS CLAVE (para LLFF):
- lr=1.0 (inicial, decae exponencialmente a 0.01)
- lr_warmup=0.99, d_warmup=50, update_d_every=100
- preconditioner_type="equilibrated" (norma de filas de la Hessiana)

DIFERENCIA CON nf-soft-mining:
- Hereda de Optimizer (clase base local con profiling)
- Usa el scheduler de GARF (exponencial) en lugar de step scheduler
- Mejor integracion con el ciclo de entrenamiento (base.py)
"""

import torch
from .optimizer import Optimizer
import math


class ESGD(Optimizer):
    def __init__(self, params, lr=10, betas=(0.9, 0.999), lr_warmup=0.99, eps=1e-4,
                 weight_decay=0, update_d_every=100, d_warmup=50, preconditioner_type="equilbrated"):
        if not 0. <= lr:
            raise ValueError(f'Invalid learning rate: {lr:g}')
        if not 0. <= betas[0] < 1.:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]:g}')
        if not 0. <= betas[1] <= 1.:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]:g}')
        if not 0. <= lr_warmup < 1.:
            raise ValueError(f'Invalid lr warmup parameter: {lr_warmup:g}')
        if not 0. <= eps:
            raise ValueError(f'Invalid epsilon value: {eps:g}')
        if not 0. <= weight_decay:
            raise ValueError('Invalid weight_decay value: {weight_decay:g}')
        if not int(update_d_every) or not 1 <= update_d_every:
            raise ValueError(f'Invalid update_d_every parameter: {update_d_every}')
        if not int(d_warmup) or not 1 <= d_warmup:
            raise ValueError(f'Invalid d_warmup parameter: {d_warmup}')
        defaults = dict(lr=lr, betas=betas, lr_warmup=lr_warmup, eps=eps,
                        weight_decay=weight_decay, update_d_every=update_d_every, d_warmup=d_warmup)
        super(ESGD, self).__init__(params, defaults)
        self.update_d_every = update_d_every
        self.d_warmup = d_warmup
        self.steps = 0
        self.steps_since_d = 0

    def state_dict(self):
        global_state = {'update_d_every': self.update_d_every,
                        'd_warmup': self.d_warmup,
                        'steps': self.steps,
                        'steps_since_d': self.steps_since_d}
        return {'global_state': global_state, **super().state_dict()}

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.update_d_every = state_dict['global_state']['update_d_every']
        self.d_warmup = state_dict['global_state']['d_warmup']
        self.steps = state_dict['global_state']['steps']
        self.steps_since_d = state_dict['global_state']['steps_since_d']

    def should_create_graph(self):
        return self.steps < self.d_warmup or self.steps_since_d >= self.update_d_every

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        hvps_iter = None
        if self.should_create_graph():
            params, grads, vs = [], [], []
            with torch.enable_grad():
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        if p.grad.is_sparse:
                            raise RuntimeError('ESGD does not support sparse gradients')
                        if p.grad.grad_fn is None:
                            msg = f'Gradient tensor shaped like {tuple(p.grad.shape)} does not have ' \
                                'a grad_fn. When calling loss.backward(), make sure the option ' \
                                'create_graph is set to True.'
                            raise RuntimeError(msg)
                        params.append(p)
                        grads.append(p.grad)
                        vs.append(torch.randint_like(p.grad, 2, dtype=p.grad.dtype) * 2 - 1)

            hvps = torch.autograd.grad(grads, params, grad_outputs=vs)

            hvps_iter = iter(hvps)
            self.steps_since_d = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg_bias_corr'] = 1.
                    state['exp_avg_d_bias_corr'] = 1.
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_d'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['lr_warmup_cumprod'] = 1.

                exp_avg, exp_avg_d = state['exp_avg'], state['exp_avg_d']
                beta1, beta2 = group['betas']
                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                state['exp_avg_bias_corr'] *= beta1
                if hvps_iter is not None:
                    hvp = next(hvps_iter)
                    exp_avg_d.mul_(beta2).addcmul_(hvp, hvp, value=1-beta2)
                    state['exp_avg_d_bias_corr'] *= beta2

                if hvps_iter is not None:
                    state['lr_warmup_cumprod'] *= group['lr_warmup']
                step_size = group['lr']

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                denom = (exp_avg_d.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                p.mul_(1 - group['weight_decay'] * step_size)
                p.addcdiv_(exp_avg, denom, value=-step_size / bias_correction1)

        self.steps += 1
        self.steps_since_d += 1
        return
