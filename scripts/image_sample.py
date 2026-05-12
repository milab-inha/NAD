"""
Run NAD sampling for multi-coil MRI reconstruction.
"""

import argparse
import json
from pathlib import Path
import time

import meddlr.ops.complex as cplx
from meddlr.forward.mri import SenseModel
import nibabel as nib
import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import create_test_data_loaders
from improved_diffusion.nn_cm import append_dims
from improved_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from noise_estimators import NOISE_ESTIMATOR_CHOICES, NoiseEstimator
from utils_hk import AverageMeter, ProgressMeter, eval_recon


def main():
    args = create_argparser().parse_args()
    if args.noise_estimator not in NOISE_ESTIMATOR_CHOICES:
        raise ValueError(
            f"unsupported noise estimator: {args.noise_estimator}. "
            f"Choose one of: {', '.join(NOISE_ESTIMATOR_CHOICES)}."
        )
    dist_util.setup_dist(which_gpu=args.which_gpu)

    exp_name = (
        f"nad_mask_{args.mask_pattern}_acc_rate_{args.acc_rate}_"
        f"steps_{args.steps}_noise_{args.noise_estimator}"
    )
    exp_dir = Path(args.output_dir) / exp_name
    recon_dir = exp_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=exp_dir)

    logger.log("creating NAD model...")
    args.timestep_respacing = str(args.steps)
    model, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_nad_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading test data...")
    data_test = create_test_data_loaders(args)
    psnr = AverageMeter("PSNR", ":.3f")
    nrmse = AverageMeter("NRMSE", ":.3f")
    ssim = AverageMeter("SSIM", ":.3f")
    progress = ProgressMeter(len(data_test), [psnr, nrmse, ssim], prefix="Test: ")

    results = []
    total_running_time = 0

    for batch_idx, data in enumerate(data_test):
        print(batch_idx, data["fname"])
        kspace_masked = data["kspace_masked"]
        kspace_masked = kspace_masked.permute((0, 3, 1, 2, 4)).view(
            -1,
            args.batch_size,
            kspace_masked.shape[1],
            kspace_masked.shape[2],
            kspace_masked.shape[4],
        )

        maps = data["maps"]
        maps = maps.permute((0, 3, 1, 2, 4, 5)).view(
            -1,
            args.batch_size,
            maps.shape[1],
            maps.shape[2],
            maps.shape[4],
            maps.shape[5],
        )

        target = data["target"]
        target = target.permute((0, 3, 1, 2, 4)).view(
            -1,
            args.batch_size,
            target.shape[1],
            target.shape[2],
        )
        target_std = data["target_std"].to(dist_util.dev()).view(-1, 1, 1, 1)

        output = {"recon": th.zeros_like(data["target"])}
        start_time = time.time()

        for batch in range(kspace_masked.shape[0]):
            with th.no_grad():
                kspace_batch = kspace_masked[batch].to(dist_util.dev()) / target_std
                maps_batch = maps[batch].to(dist_util.dev())
                mask = cplx.get_mask(kspace_batch)
                forward_op = SenseModel(maps_batch, weights=mask)

                def normal_op(x):
                    return x + forward_op(forward_op(x), adjoint=True)

                x_t = complex_x_to_real_x(sense_recon(kspace_batch, forward_op, iters=args.sense_iters))
                output_recon = denoise_iterative(
                    model,
                    x_t,
                    kspace_batch,
                    forward_op,
                    normal_op,
                    iterations=args.steps,
                    cg_iter=args.cg_iter,
                    gamma=args.gamma,
                    noise_estimator_method=args.noise_estimator,
                    noise_patch_size=args.noise_patch_size,
                )

            output["recon"][:, :, :, batch * args.batch_size : (batch + 1) * args.batch_size, ...] = (
                th.view_as_complex(output_recon.permute((0, 2, 3, 1)).contiguous())
                .permute((1, 2, 0))
                .unsqueeze(0)
                .unsqueeze(-1)
                .cpu()
            )

        running_time = time.time() - start_time
        total_running_time += running_time
        print("running time:", running_time)

        eval_result = eval_recon(
            output["recon"].to(dist_util.dev()) * target_std.view(-1, 1, 1, 1, 1).to(dist_util.dev()),
            data["target"].to(dist_util.dev()),
        )
        psnr.update(eval_result["psnr_mag"].item(), data_test.batch_size)
        nrmse.update(eval_result["nrmse"].item(), data_test.batch_size)
        ssim.update(eval_result["ssim_wang"].item(), data_test.batch_size)
        synchronize_metrics(psnr)
        synchronize_metrics(nrmse)
        synchronize_metrics(ssim)

        results.append(
            {
                "batch_idx": batch_idx,
                "fname": data["fname"][0],
                "psnr": psnr.val,
                "ssim": ssim.val,
                "nrmse": nrmse.val,
                "running_time": running_time,
            }
        )

        recon_output = np.squeeze(output["recon"].cpu().numpy(), axis=0)
        save_as_nifti(recon_output, str(recon_dir / f'{data["fname"][0]}_recon_steps_{args.steps}.nii.gz'))
        progress.display(batch_idx)

    final_result = {
        "method": "nad",
        "noise_estimator": args.noise_estimator,
        "num_steps": args.steps,
        "avg_psnr": psnr.avg,
        "std_psnr": psnr.std,
        "avg_ssim": ssim.avg,
        "std_ssim": ssim.std,
        "avg_nrmse": nrmse.avg,
        "std_nrmse": nrmse.std,
        "total_running_time": total_running_time,
        "batch_results": results,
    }
    result_file = exp_dir / f"result_steps_{args.steps}.json"
    with open(result_file, "w") as f:
        json.dump(final_result, f, indent=2)

    logger.log(f"Results saved to {result_file}")
    dist.barrier()
    logger.log("NAD sampling complete")


def synchronize_metrics(metric):
    metric_tensor = th.tensor([metric.sum, metric.count], dtype=th.float32, device=dist_util.dev())
    dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
    metric.sum = metric_tensor[0].item()
    metric.count = metric_tensor[1].item()
    metric.avg = metric.sum / metric.count


def real_x_to_complex_x(x):
    return th.view_as_complex(x.permute(0, 2, 3, 1).unsqueeze(-2).contiguous())


def complex_x_to_real_x(x):
    return th.view_as_real(x).squeeze(-2).permute(0, 3, 1, 2).contiguous()


def sense_recon(kspace_masked, forward_op, iters=30):
    image_sense = forward_op(kspace_masked, adjoint=True)
    for _ in range(iters):
        grad = forward_op(forward_op(image_sense) - kspace_masked, adjoint=True)
        step = compute_step_size(grad, forward_op)
        image_sense = image_sense - step * grad
    return image_sense


def compute_step_size(grad, forward_op, eps=1e-8):
    grad_flat = grad.view(1, -1)
    ata_grad = forward_op(forward_op(grad), adjoint=True)
    numerator = th.matmul(grad_flat, grad_flat.T)
    denominator = th.matmul(grad_flat, (grad + ata_grad).view(1, -1).T)
    return numerator / (denominator + eps)


def get_scalings(sigma):
    sigma_data = 1
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
    c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5
    return c_skip, c_out, c_in


def conjugate_gradient(operator, b, x, cg_iter=5, eps=1e-5):
    r = b - operator(x)
    p = r.clone()
    rsold = th.matmul(r.view(1, -1), r.view(1, -1).T)
    for _ in range(cg_iter):
        ap = operator(p)
        alpha = rsold / th.matmul(p.view(1, -1), ap.view(1, -1).T)
        x = x + alpha * p
        r = r - alpha * ap
        rsnew = th.matmul(r.view(1, -1), r.view(1, -1).T)
        if th.abs(th.sqrt(rsnew)) < eps:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x


def denoise(model, x_t, sigmas):
    c_skip, c_out, c_in = [append_dims(x, x_t.ndim) for x in get_scalings(sigmas)]
    model_output = model(c_in * x_t, 1000 * 0.25 * th.log(sigmas + 1e-44))
    return c_out * model_output + c_skip * x_t


def denoise_iterative(
    model,
    x_t,
    kspace,
    forward_op,
    normal_op,
    iterations=50,
    cg_iter=5,
    gamma=1.0,
    noise_estimator_method="pca",
    noise_patch_size=8,
):
    noise_estimator = NoiseEstimator(patch_size=noise_patch_size)
    for step in range(iterations):
        x_t_noise = noise_estimator.estimate(x_t, method=noise_estimator_method).to(dist_util.dev())
        x_0 = denoise(model, x_t, x_t_noise)
        b_cg = real_x_to_complex_x(x_0) + forward_op(kspace, adjoint=True)
        x_0_cg = complex_x_to_real_x(
            conjugate_gradient(normal_op, b_cg, real_x_to_complex_x(x_0), cg_iter=cg_iter)
        )
        noise_factor = gamma * (3 - (3 * step) / iterations)
        x_t = x_0_cg + th.randn_like(x_0_cg) * x_t_noise.view(-1, 1, 1, 1) * noise_factor
    return x_0_cg


def save_as_nifti(data, filepath):
    magnitude_data = np.abs(data)
    magnitude_data = np.squeeze(magnitude_data).transpose(2, 0, 1)
    nib.save(nib.Nifti1Image(magnitude_data, affine=np.eye(4)), filepath)


def create_argparser():
    defaults = dict(
        batch_size=4,
        model_nad_path="/path/to/model.pt",
        output_dir="output",
        steps=50,
        sense_iters=30,
        cg_iter=5,
        gamma=1.0,
        noise_estimator="pca",
        noise_patch_size=8,
        seed=42,
        dropout=0.0,
        data_path_npy="/path/to/skm-tea/data_npy",
        data_path="/path/to/skm-tea",
        seg_mask_path="/path/to/skm-tea",
        acc_rate=4,
        mask_pattern="random",
        slice_dim=2,
        echo=0,
        sample_rate_train=1,
        sample_rate_test=1,
        data_train_num_worker=4,
        data_eval_num_worker=0,
        which_gpu=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
