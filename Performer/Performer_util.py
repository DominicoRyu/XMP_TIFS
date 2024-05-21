import math
from operator import mul
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptFormer(nn.Module):
    def __init__(self, in_dim, bottle_dim, adapter_scalar = "0.1", dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    @property
    def dtype(self):
        return self.ln.weight.dtype

    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        x = x * self.scale
        return x

def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args

class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route = {}, config = None):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x

class SequentialSequence_adapt(nn.Module):
    def __init__(self, layers, args_route={}, config=None):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route
        self.adapter = AdaptFormer(in_dim=config["d_model"],
                                   bottle_dim=config["ffn_num"],
                                   adapter_scalar=config["ffn_adapter_scalar"])

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            residual = x
            x = x + g(x, **g_args) + self.adapter(residual)
        return x


class SequentialSequence_vpt(nn.Module):
    def __init__(self, layers, args_route={}, config=None):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route

        # VPT 항상 활성화
        self.vpt_num = config.get("vpt_num", 1)
        self.vpt_prompts = nn.ParameterList([nn.Parameter(torch.zeros(1, self.vpt_num, config["d_model"])) for _ in range(len(self.layers))])

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        for idx, ((f, g), (f_args, g_args)) in enumerate(layers_and_args):
            vpt_prompt = self.vpt_prompts[idx].expand(x.size(0), -1, -1)
            x = torch.cat((vpt_prompt, x), dim=1)

            x = x + f(x, **f_args)  # self-attention
            x = x + g(x, **g_args)  # feed-forward

            x = x[:, self.vpt_num:, :]
        return x
