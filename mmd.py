"""MMD losses reused by the online UDA-DDA experiment."""

from __future__ import annotations

import torch


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    l2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma is not None:
        bandwidth = torch.as_tensor(fix_sigma, device=source.device, dtype=source.dtype)
    else:
        denom = max(n_samples**2 - n_samples, 1)
        bandwidth = torch.sum(l2_distance.detach()) / denom
    bandwidth = torch.clamp(bandwidth, min=1e-6)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    return sum(torch.exp(-l2_distance / bw) for bw in bandwidth_list)


def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    if source.numel() == 0 or target.numel() == 0:
        return source.new_tensor(0.0)
    batch_size_s = int(source.size(0))
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)
    xx = kernels[:batch_size_s, :batch_size_s]
    yy = kernels[batch_size_s:, batch_size_s:]
    xy = kernels[:batch_size_s, batch_size_s:]
    yx = kernels[batch_size_s:, :batch_size_s]
    return torch.mean(xx) + torch.mean(yy) - torch.mean(xy) - torch.mean(yx)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Compatibility alias.

    The online buffers can be smaller than the source batch, so this implementation
    intentionally uses the non-accelerated form that supports unequal batch sizes.
    """
    return mmd_rbf_noaccelerate(source, target, kernel_mul, kernel_num, fix_sigma)
