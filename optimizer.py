import math, torch

class DecoupledLion(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-4,beta1=0.9,beta2=0.99,weight_decay=0.0):
        defaults=dict(lr=lr,beta1=beta1,beta2=beta2,weight_decay=weight_decay)
        super().__init__(params,defaults)
    @torch.no_grad()
    def step(self,closure=None):
        loss=closure() if closure else None
        for g in self.param_groups:
            lr,bet1,bet2,wd=g['lr'],g['beta1'],g['beta2'],g['weight_decay']
            for p in g['params']:
                if not p.grad: continue
                d=p.grad; state=self.state[p]
                if not state: state['m']=torch.zeros_like(p)
                m=state['m']
                m.mul_(bet1).add_(d,alpha=1-bet1)
                update=d.add(m,alpha=bet2)
                if wd: p.add_(p,alpha=-lr*wd)
                p.add_(update.sign(),alpha=-lr)
        return loss

def make_optimizer(params,lr,weight_decay,warmup_steps,total_steps):
    adam=torch.optim.AdamW(params,lr=lr,betas=(0.9,0.95),weight_decay=weight_decay,eps=1e-8)
    def schedule(step):
        if step<warmup_steps: return step/warmup_steps
        progress=(step-warmup_steps)/(total_steps-warmup_steps)
        return 0.5*(1+math.cos(math.pi*progress))
    # wrapper object
    class _Opt:
        def __init__(self): self._lion=DecoupledLion(params,lr=lr*0.5,weight_decay=weight_decay); self.step_num=0
        def zero_grad(self): adam.zero_grad(); self._lion.zero_grad()
        def step(self): self.step_num+=1
            lr_scale=schedule(self.step_num); for g in adam.param_groups: g['lr']=lr*lr_scale
            (adam if self.step_num<warmup_steps*2 else self._lion).step()
    return _Opt()