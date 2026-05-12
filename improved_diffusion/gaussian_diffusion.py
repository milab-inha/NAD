"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
import meddlr.ops.complex as cplx
from meddlr.forward.mri import SenseModel
from functools import partial
import matplotlib.pyplot as plt

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_sample_measurement(self, x_start, y_start, t, A):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        z = th.randn_like(x_start)
        Az = A(real_x_to_complex_x(z))

        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, y_start.shape) * y_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, y_start.shape)
            * Az
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self, model, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]


    def dds_sample(
        self,
        model,
        x,
        t,
        A,
        Afull,
        Acg_fn,
        kspace_part,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        x_0_dc = self.data_consistency(kspace=kspace_part, A=A, Acg_fn=Acg_fn, x0_t=out["pred_xstart"])
        # print(out["pred_xstart"].min(), out["pred_xstart"].max(), x_0_dc.min(), x_0_dc.max())
        mean_pred = (
            x_0_dc * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        # print(sample.max(), sample.min())
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def sure_sample(
        self,
        model,
        x,
        t,
        t_0,
        A,
        Acg_fn,
        kspace_part,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        x_0_dc = self.data_consistency(kspace=kspace_part, A=A, Acg_fn=Acg_fn, x0_t=out["pred_xstart"])
        x_0_ot = self.sure_estimate_and_gradient(model, x_0_dc, t_0)
        # print(out["pred_xstart"].min(), out["pred_xstart"].max(), x_0_dc.min(), x_0_dc.max())
        mean_pred = (
            x_0_ot * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def score_sample(
        self,
        model,
        x,
        t,
        A,
        Afull,
        Acg_fn,
        kspace_part,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        noisy_measurement = self.q_sample_measurement(x_start=x, y_start=kspace_part, t=t, A=A)
        # update = complex_x_to_real_x(A(A(real_x_to_complex_x(x))-noisy_measurement, adjoint=True))
        # print(th.norm(update), th.norm(x))
        # x = x - update
        mask = A.weights
        ld = 0.5
        term1 = ld * A(noisy_measurement, adjoint=True)
        term2 = (1 - ld) * Afull(A(real_x_to_complex_x(x)), adjoint=True)
        term3 = Afull((1 - mask) * Afull(real_x_to_complex_x(x)), adjoint=True)
        # update = complex_x_to_real_x(AT(A(real_x_to_complex_x(x))-noisy_measurement))
        # print(th.norm(update), th.norm(x))
        # x = x - update
        x = complex_x_to_real_x(term1 + term2 + term3)
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        data,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        method='',
        sure_param=0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        # target = data["target"].unsqueeze(-1) / data['target_std']
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            data,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            method=method,
            sure_param=sure_param,
        ):
            final = sample
        # diff = (final["sample"] - complex_x_to_real_x(target)) ** 2
        # print("mse: ", diff.sum(dim=(1, 2, 3)))
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        data,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        method='',
        sure_param=0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        kspace_part = data["kspace_part"]
        target_std = data['target_std']
        maps = data["maps"]
        kspace_part /= target_std
        mask = cplx.get_mask(kspace_part)
        A = SenseModel(maps, weights=mask)
        Afull = SenseModel(maps, weights=1)
        def Acg_noise(x, gamma):
            return x + gamma * A(A(x), adjoint=True)
        # def Acg(x):
        #     return A(A(x), adjoint=True)
        Acg_fn = partial(Acg_noise, gamma=1)
        data_args = dict(
            kspace_part=kspace_part, A=A, Acg_fn=Acg_fn, Afull=Afull,
        )
        scale_proj = complex_x_to_real_x(A(A(real_x_to_complex_x(th.ones_like(img))), adjoint=True))
        if method == 'ssd':
            # SSD parameters
            t_0 = min(550, self.num_timesteps - 1)
            if self.num_timesteps < 15:
                portion = 0.4
            else:
                portion = 0.3
            S_inv = max(1, int(portion * len(indices)))
            S_gen = len(indices) - S_inv
            eta_inversion = 0.4

            # Step 1: DA Inversion
            x = complex_x_to_real_x(A(kspace_part, adjoint=True))
            tau = np.linspace(0, t_0, S_inv + 1)

            for i in range(S_inv):
                t = th.full((shape[0],), int(tau[i]), device=device, dtype=th.long)
                t_next = th.full((shape[0],), int(tau[i + 1]), device=device, dtype=th.long)

                with th.no_grad():
                    eps = model(x, self._scale_timesteps(t))
                    x_0 = self._predict_xstart_from_eps(x, t, eps)

                    alpha = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
                    alpha_next = _extract_into_tensor(self.alphas_cumprod, t_next, x.shape)
                    beta = th.clamp(1 - alpha / alpha_next, min=1e-5)

                    noise = th.randn_like(x)
                    x = th.sqrt(alpha_next + 1e-8) * x_0 + \
                        th.sqrt(th.clamp(1 - alpha_next - eta_inversion * beta, min=1e-8)) * eps + \
                        th.sqrt(th.clamp(eta_inversion * beta, min=1e-8)) * noise

                    if th.isnan(x).any():
                        print(f"NaN detected in x at step {i} of inversion")
                        break

                yield {"sample": x, "pred_xstart": x_0}

            # Step 2: Generation
            for i in range(S_gen):
                t = th.full((shape[0],), int(t_0 * (S_gen - i - 1) / S_gen), device=device, dtype=th.long)

                with th.no_grad():
                    out = self.p_sample(
                        model,
                        x,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                    )

                    # Back projection
                    x_0 = out["pred_xstart"]
                    x_0_complex = real_x_to_complex_x(x_0)
                    kspace_pred = A(x_0_complex, adjoint=False)
                    kspace_error = kspace_part - kspace_pred
                    x_0_update = A(kspace_error, adjoint=True)
                    x_0_bp = x_0_complex + x_0_update
                    x_0_bp = complex_x_to_real_x(x_0_bp)

                    # DDIM-style update
                    eps = self._predict_eps_from_xstart(x, t, x_0_bp)
                    alpha = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
                    alpha_next = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
                    sigma = eta * th.sqrt((1 - alpha_next) / (1 - alpha + 1e-8)) * th.sqrt(
                        1 - alpha / (alpha_next + 1e-8))

                    noise = th.randn_like(x)
                    mean_pred = x_0_bp * th.sqrt(alpha_next + 1e-8) + \
                                th.sqrt(th.clamp(1 - alpha_next - sigma ** 2, min=1e-8)) * eps
                    x = mean_pred + sigma * noise

                    if th.isnan(x).any():
                        print(f"NaN detected in x at step {i} of generation")
                        break

                yield {"sample": x, "pred_xstart": x_0_bp}

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            t_0 = th.tensor([sure_param] * shape[0], device=device)
            if method=='sure':
                with th.no_grad():
                    out = self.sure_sample(
                        model,
                        img,
                        t,
                        t_0,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                        **data_args,
                    )
                    yield out
                    img = out["sample"]

            elif method=='nad':
                with th.no_grad():
                    out = self.nad_sample(
                        model,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                        **data_args,
                    )
                    yield out
                    img = out["sample"]

            elif method=='dds':
                with th.no_grad():
                    out = self.dds_sample(
                        model,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                        **data_args,
                    )
                    yield out
                    img = out["sample"]


            elif method == 'dps':
                img = img.requires_grad_()
                out = self.p_sample(
                    model=model,
                    x=img,
                    t=t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                img, distance = self.measurement_cond_fn_dps(x_t=out['sample'],
                                                    measurement=kspace_part,
                                                    noisy_measurement=None,
                                                    x_prev=img,
                                                    x_0_hat=out['pred_xstart'],
                                                    A=A,
                                                    scale=1,
                                                    )
                img = img.detach_()
                out["sample"] = img
                yield out

            elif method=='mcg':
                img = img.requires_grad_()
                out = self.p_sample(
                    model=model,
                    x=img,
                    t=t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                noisy_measurement = self.q_sample(x_start=kspace_part, t=t)
                img, distance = self.measurement_cond_fn_mcg(x_t=out['sample'],
                                                         measurement=kspace_part,
                                                         noisy_measurement=noisy_measurement,
                                                         x_prev=img,
                                                         x_0_hat=out['pred_xstart'],
                                                         A=A,
                                                         scale=1,
                                                         scale_proj=scale_proj,
                                                         )
                img = img.detach_()
                out["sample"] = img
                yield out

            elif method == 'score':
                with th.no_grad():
                    out = self.score_sample(
                        model,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                        **data_args,
                    )
                    yield out
                    img = out["sample"]

            elif method == 'diffuse':
                with th.no_grad():
                    out = self.coarse_to_fine_sample(
                        model,
                        shape,
                        data,
                        noise=noise,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        device=device,
                        progress=progress,
                    )
                    yield out

    def get_adaptive_c2f_params(self):
        # Set T to the total number of timesteps
        T = self.num_timesteps

        # Set N (number of coarse samples) to be approximately sqrt(T/2)
        N = max(1, int(math.sqrt(T / 2)))

        # Set k (steps per coarse sample) to be approximately T / (2N)
        k = max(1, T // (2 * N))

        # Adjust N to ensure N * k is close to T/2
        N = max(1, T // (2 * k))

        # Set T_refine to be the remaining steps
        T_refine = T - (N * k)

        return T, k, N, T_refine

    def coarse_to_fine_sample(
            self,
            model,
            shape,
            data,
            noise=None,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
    ):
        if device is None:
            device = next(model.parameters()).device

        kspace_part = data["kspace_part"]
        target_std = data['target_std']
        maps = data["maps"]
        kspace_part /= target_std
        mask = cplx.get_mask(kspace_part)
        A_full = SenseModel(maps, weights=None)
        A = SenseModel(maps, weights=mask)

        # Get adaptive parameters
        T, k, N, T_refine = self.get_adaptive_c2f_params()

        # Coarse sampling
        coarse_samples = []
        for _ in range(N):
            img = th.randn(*shape, device=device) if noise is None else noise
            for i in range(T-1, 0, -int(T/k)):
                t = th.tensor([i] * shape[0], device=device)
                out = self.p_sample(
                    model=model,
                    x=img,
                    t=t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                kspace_filled = (1 - mask) * A_full(real_x_to_complex_x(out['sample']))
                y_full_est = kspace_filled + self.q_sample_measurement(x_start=out['sample'], y_start=kspace_part,t=t,A=A)
                img = complex_x_to_real_x(A_full(y_full_est, adjoint=True))
            coarse_samples.append(img)

        # Average coarse samples
        y_avg = th.stack(coarse_samples).mean(dim=0)

        # Refinement
        for i in range(T_refine, 0, -1):
            t = th.tensor([i] * shape[0], device=device)
            out = self.p_sample(
                model=model,
                x=y_avg,
                t=t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            kspace_filled = (1 - mask) * A_full(real_x_to_complex_x(out['sample']))
            y_full_est = kspace_filled + kspace_part
            y_avg = complex_x_to_real_x(A_full(y_full_est, adjoint=True))

        return {'sample': y_avg}

    def measurement_cond_fn_dps(
            self,
            x_t,
            measurement,
            noisy_measurement,
            x_prev,
            x_0_hat,
            A,
            scale,
    ):
        # difference = complex_x_to_real_x(A(measurement, adjoint=True)) - x_0_hat
        difference = measurement - A(real_x_to_complex_x(x_0_hat))
        norm = th.linalg.norm(difference)
        norm_grad = th.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * scale
        return x_t, norm

    def measurement_cond_fn_mcg(
            self,
            x_t,
            measurement,
            noisy_measurement,
            x_prev,
            x_0_hat,
            A,
            scale,
            scale_proj,
    ):
        difference = A(measurement - A(real_x_to_complex_x(x_0_hat)), adjoint=True)
        norm = th.linalg.norm(difference)
        norm_grad = th.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * scale
        proj = A(A(real_x_to_complex_x(x_t)) - noisy_measurement, adjoint=True)
        x_t = x_t - complex_x_to_real_x(proj) / th.norm(proj)
        return x_t, norm

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=False, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=False, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }

    def CG(self, A, b, x, n_inner=5, eps=1e-5):
        r = b - A(x)
        p = r.clone()
        rsold = th.matmul(r.view(1, -1), r.view(1, -1).T)
        for i in range(n_inner):
            Ap = A(p)
            denom = th.matmul(p.view(1, -1), Ap.view(1, -1).T) + 1e-8
            a = rsold / denom
            x = x + a * p
            r = r - a * Ap

            rsnew = th.matmul(r.view(1, -1), r.view(1, -1).T)
            if th.abs(th.sqrt(rsnew)) < eps:
                break
            p = r + (rsnew / (rsold+1e-8) ) * p
            rsold = rsnew
        return x

    def GD(self, A, y, x):
        grad = A(A(x) - y, adjoint=True)
        step_size = th.matmul(grad.view(1, -1), grad.view(1, -1).T) / th.matmul(grad.view(1, -1), (grad + A(A(grad), adjoint=True)).view(1, -1).T)
        # print('dc grad norm: ', th.norm(grad), ' step size: ', step_size)
        x = x - step_size * grad
        # print('update norm: ', th.norm(th.view_as_real(step_size * grad).reshape(tuple(y.size())[0:3] + (2,)).permute(0, 3, 1, 2)))
        return x
    def data_consistency(self, kspace, A, Acg_fn, x0_t):
        dims = tuple(kspace.size())
        x0_t = th.view_as_complex(x0_t.permute(0, 2, 3, 1).reshape(dims[0:3] + (1, 2)).contiguous())
        bcg = x0_t + A(kspace, adjoint=True)
        # bcg = A(kspace, adjoint=True)
        x0_t_hat = self.CG(Acg_fn, bcg, x0_t)
        # x0_t_hat = self.GD(A=A, y=kspace, x=x0_t)
        return th.view_as_real(x0_t_hat).reshape(dims[0:3] + (2,)).permute(0, 3, 1, 2)

    def sure_estimate_and_gradient(self, model, x_0, t_0, epsilon=1e-3):
        s_in = x_0.new_ones([x_0.shape[0]])
        epsilon = x_0.max() / 1e4
        b = th.randn(*x_0.shape, device=x_0.device)
        with th.enable_grad():
            x_in = x_0.detach().requires_grad_(True)
            x_0_hat = self.p_mean_variance(model, x_in, t_0)['pred_xstart']
            sigma_squared = th.mean((x_0_hat - x_in) ** 2)
            perturbations = epsilon * b
            perturbed_output = self.p_mean_variance(model, x_in + perturbations, t_0)['pred_xstart']
            diff = (perturbed_output - x_0_hat).contiguous().view(-1, 1)
            tr_J = th.matmul(b.view(1, -1), diff) / epsilon
            sure = sigma_squared * tr_J
            grad_sure = th.autograd.grad(sure, x_in)[0]
        return x_0 - 0.5*grad_sure

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def real_x_to_complex_x(x):
    batch, channel, h, w = x.shape
    dims = tuple((batch,h,w,8))
    x = th.view_as_complex(x.permute(0, 2, 3, 1).reshape(dims[0:3] + (1, 2)).contiguous())
    return x

def complex_x_to_real_x(x):
    batch, h, w, _ = x.shape
    dims = tuple((batch, h, w, 8))
    return th.view_as_real(x).reshape(dims[0:3] + (2,)).permute(0, 3, 1, 2)

def complex_y_to_real_y(y):
    y = th.view_as_real(y)
    batch, h, w, coils, _ = y.shape
    y = y.view(batch, h, w, -1)
    y = y.permute(0, 3, 1, 2)
    return y

def real_y_to_complex_y(y):
    # y: real-valued tensor [batch, 2*coils, height, width] from U-Net
    batch, channels, h, w = y.shape
    coils = channels // 2
    y = y.permute(0, 2, 3, 1)
    y = y.view(batch, h, w, coils, 2)
    y_complex = th.view_as_complex(y.contiguous())
    return y_complex

