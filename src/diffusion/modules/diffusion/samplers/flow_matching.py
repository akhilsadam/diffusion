# samplers file
from .cache import _Cache
import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['TRITON_BACKEND'] = 'cuda'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo as dynamo

@dynamo.disable
def dot(a,b):
    return sum([a[i]*b[i] for i in range(len(a))])
    
class Euler(nn.Module):
    def reset(self):
        pass
    @torch.compile
    def step(self, net, x, i, t, dt, **kwargs):
        v_pred = net.vel(net.denoise(x, t[i], **kwargs), x, t[i])
        x = x + dt[i] * v_pred
        return x

class AB2(nn.Module):
    def __init__(self, level=2):
        super().__init__()
        self.cache = _Cache(
            level,
            ['v', 'h'],
            lambda table: dot(table['v'], self.coeff(table['h']))
        )
        self.cache.reset()
        
    def reset(self):
        self.cache.reset()

    def coeff(self, h): # to be extended
        # h[0] = current dt (h_n)
        # h[1] = previous dt (h_{n-1})
        
        if len(h) < 2:
            return [1.0]  # fallback to Euler
        
        hn = h[0]
        hnm1 = h[1]
        
        r = hnm1 / hn
        
        b0 = 1 + 1/(2*r)
        b1 = -1/(2*r)
        
        return [b0, b1]
        
    @torch.compile
    def step(self, net, x, i, t, dt, **kwargs):
        
        v_pred = net.vel(net.denoise(x, t[i], **kwargs), x, t[i])
        v_pred_ab = self.cache({'v': v_pred, 'h': dt[i]})
        
        x = x + dt[i] * v_pred_ab # AB2 update
        
        return x

class AB2CN(AB2):
    def __init__(self, level=2):
        super().__init__(level=level)
        
    @torch.compile
    def step(self, net, x, i, t, dt, **kwargs):
        
        v_pred = net.vel(net.denoise(x, t[i], **kwargs), x, t[i])
        v_pred_ab = self.cache({'v': v_pred, 'h': dt[i]})
        
        x_plus_1 = x + dt[i] * v_pred_ab # AB2 update
        v_plus_1 = net.vel(net.denoise(x_plus_1, t[i+1], **kwargs), x, t[i]) 
        # can't take forward v since that doesn't exist at final time
        # anyways, we want reverse v...so we can assume time is same

        v_avg = (v_plus_1 + v_pred_ab) / 2

        x = x + dt[i] * v_avg 
        
        return x
     
    
samplers = {
    'Euler': Euler,
    'AB2': AB2,
    'AB2CN': AB2CN,
}