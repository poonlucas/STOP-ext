"""
Adapted from: https://pytorch.org/docs/1.6.0/_modules/torch/optim/adam.html
"""
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import _dispatch_sqrt, _get_value


class CAdam(Optimizer):
    def __init__(self, params, lr: 1e-3, betas: (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(CAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]

                    # Lazy state initialization
                    if len(state) == 0:
                        # note(crcrpar): [special device hosting for step]
                        # Deliberately host `step` on CPU if both capturable and fused are off.
                        # This is because kernel launches are costly on CUDA and XLA.
                        state['step'] = torch.tensor(0.0, dtype=torch.float32)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    state_steps.append(state['step'])

                    for i, param in enumerate(params_with_grad):
                        grad = grads[i]
                        exp_avg = exp_avgs[i]
                        exp_avg_sq = exp_avg_sqs[i]
                        step_t = state_steps[i]

                        step_t += 1

                        if group['weight_decay'] != 0:
                            grad = grad.add(param, alpha=group['weight_decay'])

                        # Decay the first and second moment running average coefficient
                        exp_avg.lerp_(grad, 1 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                        step = _get_value(step_t)

                        bias_correction1 = 1 - beta1 ** step
                        bias_correction2 = 1 - beta2 ** step

                        step_size = group['lr'] / bias_correction1

                        bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

                        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])

                        param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
