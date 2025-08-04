import torch
import autort

autort.export(name='rmsprop_bf16', ir="""
    weight[N].set(weight[N].float() - extra_f32(strs.lr) * grad[N].float() * (avg[N].float() + 1e-8).rsqrt())
""", inputs=["weight=bfloat16[N]", "grad=bfloat16[N]", "avg=bfloat16[N]"])


def apply_gradient(param, square_avg, grad, group):
    alpha = group.get('alpha', 0.99)
    square_avg.mul_(alpha).addcmul_(grad, grad, value=1.0 - alpha)
    autort.ops.rmsprop_bf16(square_avg.flatten(), grad.flatten(), param.data.flatten(), extra=[group['lr'],])
    # param.data.addcdiv_(grad, square_avg.sqrt().add_(group['eps']), value=-group['lr'])


class RMSprop(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, alpha=0.99, eps=1e-8):
        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        super(RMSprop, self).__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                p.state = torch.zeros_like(p.data, memory_format=torch.preserve_format)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'step' not in state:
                    state['step'] = 0
                    # p.state.zero_()

                state['step'] += 1
                apply_gradient(p, p.state, grad, group)

        return loss

