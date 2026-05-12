from enum import Enum
import math

import meddlr.metrics.functional as mF
from meddlr.ops import complex as cplx
import torch


def eval_recon(output, target):
    if output.shape[1] == 2:
        output = torch.view_as_complex(output.permute((0, 2, 3, 1)).contiguous()).unsqueeze(-1)

    abs_error = cplx.abs(output - target)
    l1 = torch.mean(abs_error)
    output_cf = cplx.channels_first(output.detach()).contiguous()
    target_cf = cplx.channels_first(target.detach()).contiguous()
    psnr_mag = mF.psnr(output_cf, target_cf, im_type="magnitude")
    nrmse = mF.nrmse(output_cf, target_cf, im_type="magnitude")
    ssim_wang = mF.ssim(output_cf, target_cf, method="wang", im_type="magnitude").mean()

    return {
        "l1": l1,
        "nrmse": nrmse.mean(),
        "psnr_mag": psnr_mag.mean(),
        "ssim_wang": ssim_wang,
        "ssim_loss_wang": 1 - ssim_wang,
    }


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    """Computes and stores the average, current value, and standard deviation."""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.std = 0
        self.sum = 0
        self.sum_sq = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.sum_sq += val * val * n
        self.count += n
        self.avg = self.sum / self.count
        self.std = math.sqrt((self.sum_sq / self.count) - (self.avg * self.avg)) if self.count > 1 else 0

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "} +/- {std" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            return ""
        if self.summary_type is Summary.AVERAGE:
            return f"{self.name} {self.avg:.3f} +/- {self.std:.3f}"
        if self.summary_type is Summary.SUM:
            return f"{self.name} {self.sum:.3f}"
        if self.summary_type is Summary.COUNT:
            return f"{self.name} {self.count:.3f}"
        raise ValueError(f"invalid summary type {self.summary_type!r}")


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = ["*"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
