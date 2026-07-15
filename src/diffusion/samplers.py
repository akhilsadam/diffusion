import torch
import torch.nn as nn
import torch.nn.functional as F


def dot(a, b):
    """Compute dot product of two lists."""
    return sum([a[i] * b[i] for i in range(len(a))])


class _Cache:
    """Caching utility for multi-step methods."""

    def __init__(self, level, cache_labels, optable):
        self.level = level
        self.keys = cache_labels
        self.optable = optable
        self.reset()

    def update(self, items):
        for key, val in items.items():
            self.cache[key] = [val, *self.cache[key][: self.level - 1]]

    def reset(self):
        self.cache = {k: [] for k in self.keys}

    def __call__(self, items):
        self.update(items)
        return self.optable(self.cache)


class Euler(nn.Module):
    """Euler method for ODE integration."""

    @torch.compile
    def step(self, net, x, i, t, dt):
        v_pred = net.vel(net.denoise(x, t[i]), x, t[i])
        x = x + dt[i] * v_pred
        return x

    def reset(self):
        pass


class AB2(nn.Module):
    """Adams-Bashforth 2-step method."""

    def __init__(self, level=2):
        super().__init__()
        self.cache = _Cache(level, ["v", "h"], lambda table: dot(table["v"], self.coeff(table["h"])))
        self.cache.reset()

    def reset(self):
        self.cache.reset()

    def coeff(self, h):
        if len(h) < 2:
            return [1.0]

        hn = h[0]
        hnm1 = h[1]
        r = hnm1 / hn

        b0 = 1 + 1 / (2 * r)
        b1 = -1 / (2 * r)

        return [b0, b1]

    def step(self, net, x, i, t, dt):
        v_pred = net.vel(net.denoise(x, t[i]), x, t[i])
        v_pred_ab = self.cache({"v": v_pred, "h": dt[i]})
        x = x + dt[i] * v_pred_ab
        return x


class AB5(AB2):
    """Adams-Bashforth 5-step method."""

    def coeff(self, h):
        h_dict = {
            1: [1],
            2: [3 / 2, -1 / 2],
            3: [23 / 12, -4 / 3, 5 / 12],
            4: [55 / 24, -59 / 24, 37 / 24, -3 / 8],
            5: [1901 / 720, -1387 / 360, 109 / 30, -637 / 360, 251 / 720],
        }
        return h_dict[min(len(h), 5)]


class AB2CN(AB2):
    """Adams-Bashforth 2-step Crank-Nicolson hybrid."""

    def __init__(self, level=2):
        super().__init__(level=level)

    def step(self, net, x, i, t, dt):
        v_pred = net.vel(net.denoise(x, t[i]), x, t[i])
        v_pred_ab = self.cache({"v": v_pred, "h": dt[i]})

        x_plus_1 = x + dt[i] * v_pred_ab
        v_plus_1 = net.vel(net.denoise(x_plus_1, t[i + 1]), x, t[i])

        v_avg = (v_plus_1 + v_pred_ab) / 2
        x = x + dt[i] * v_avg

        return x
