"""Conditional MMD loss used by UDA-DDA."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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


def cmmd(
    source,
    target,
    s_label,
    t_label,
    num_classes=3,
    kernel_mul=2.0,
    kernel_num=5,
    fix_sigma=None,
):
    """Class-conditional MMD.

    Labels are integer class ids. Empty target selections return a zero tensor so
    S1 can skip CMMD when no pseudo labels pass the confidence threshold.
    """
    if source.numel() == 0 or target.numel() == 0:
        return source.new_tensor(0.0)

    s_label = s_label.to(device=source.device, dtype=torch.long).view(-1)
    t_label = t_label.to(device=source.device, dtype=torch.long).view(-1)
    s_one_hot = F.one_hot(s_label, num_classes=num_classes).to(dtype=source.dtype)
    t_one_hot = F.one_hot(t_label, num_classes=num_classes).to(dtype=source.dtype)

    batch_size_s = source.size(0)
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)
    xx = kernels[:batch_size_s, :batch_size_s]
    yy = kernels[batch_size_s:, batch_size_s:]
    xy = kernels[:batch_size_s, batch_size_s:]
    yx = kernels[batch_size_s:, :batch_size_s]

    loss_xx = torch.mean(torch.mm(s_one_hot, s_one_hot.t()) * xx)
    loss_yy = torch.mean(torch.mm(t_one_hot, t_one_hot.t()) * yy)
    loss_xy = torch.mean(torch.mm(s_one_hot, t_one_hot.t()) * xy)
    loss_yx = torch.mean(torch.mm(t_one_hot, s_one_hot.t()) * yx)
    return (loss_xx + loss_yy - loss_xy - loss_yx) / float(num_classes)
