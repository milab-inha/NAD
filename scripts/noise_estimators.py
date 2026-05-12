import numpy as np
import pywt
from scipy.optimize import minimize
from skimage.metrics import mean_squared_error
from skimage.restoration import denoise_tv_chambolle
import torch
import torch.nn.functional as F


NOISE_ESTIMATOR_CHOICES = (
    "pca",
    "mri_wavelet",
    "tv",
    "block_adaptive_gaussian",
    "adaptive_wavelet",
    "scale_invariance",
)


def gaussian_kernel(kernel_size=5, sigma=1.0, channels=1):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2))
        * np.exp(
            -(
                (x - (kernel_size - 1) / 2) ** 2
                + (y - (kernel_size - 1) / 2) ** 2
            )
            / (2 * sigma**2)
        ),
        (kernel_size, kernel_size),
    )
    kernel /= np.sum(kernel)
    kernel = torch.tensor(kernel, dtype=torch.float32)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    return kernel.repeat(channels, 1, 1, 1)


def apply_gaussian_blur(image, kernel_size=5, sigma=1.0):
    channels = image.shape[0]
    kernel = gaussian_kernel(kernel_size, sigma, channels).to(image.device)
    image = image.unsqueeze(0)
    blurred_image = F.conv2d(image, kernel, padding=kernel_size // 2, groups=channels)
    return blurred_image.squeeze(0)


class NoiseEstimator:
    def __init__(self, patch_size=8):
        self.patch_size = patch_size

    def estimate(self, batch, method="pca"):
        if method == "pca":
            return self.estimate_noise_level_pca(batch)
        if method == "mri_wavelet":
            return self.estimate_noise_level_mri_wavelet(batch)
        if method == "tv":
            return self.estimate_noise_level_tv(batch)
        if method == "block_adaptive_gaussian":
            return self.estimate_noise_level_block_adaptive_gaussian(batch)
        if method in ("adaptive_wavelet", "adap_wavelet"):
            return self.estimate_noise_level_adaptive_wavelet(batch)
        if method == "scale_invariance":
            return self.estimate_noise_level_scale_invariance(batch)
        raise ValueError(
            f"unsupported noise estimator: {method}. "
            f"Choose one of: {', '.join(NOISE_ESTIMATOR_CHOICES)}."
        )

    def mad(self, arr):
        arr = arr.flatten()
        med = torch.median(arr)
        return torch.median(torch.abs(arr - med)) / 0.6745

    def wavelet_denoise(self, image, wavelet="db1", mode="soft", level=1):
        coeffs = pywt.wavedec2(image.detach().cpu().numpy(), wavelet, level=level)
        sigma_est = self.mad(torch.as_tensor(self._flatten_detail_coeffs(coeffs[-1]), device=image.device))
        threshold = sigma_est * np.sqrt(2 * np.log(image.numel()))
        coeffs = list(coeffs)
        coeffs[1:] = [
            tuple(pywt.threshold(c, value=threshold.item(), mode=mode) for c in detail)
            for detail in coeffs[1:]
        ]
        return torch.as_tensor(pywt.waverec2(coeffs, wavelet), device=image.device)

    def estimate_noise_level_wavelet(self, image, wavelet="db1", level=1):
        coeffs = pywt.wavedec2(image.detach().cpu().numpy(), wavelet, level=level)
        detail_coeffs = torch.as_tensor(self._flatten_detail_coeffs(coeffs[-1]), device=image.device)
        return self.mad(detail_coeffs)

    def extract_patches(self, image, patch_size=None, stride=1):
        patch_size = patch_size or self.patch_size
        unfold = F.unfold(image.unsqueeze(0).unsqueeze(0), kernel_size=patch_size, stride=stride)
        return unfold.squeeze(0).transpose(0, 1)

    def pca_torch(self, x):
        x_centered = x - x.mean(dim=0)
        cov_matrix = torch.mm(x_centered.T, x_centered) / (x_centered.shape[0] - 1)
        eigenvalues, _ = torch.linalg.eigh(cov_matrix)
        return eigenvalues

    def estimate_noise_level_from_eigenvalues(self, eigenvalues):
        values = eigenvalues.detach().cpu().tolist()
        while len(values) > 1:
            mean = sum(values) / len(values)
            median = float(np.median(values))
            if abs(mean - median) < 1e-6:
                break
            values.remove(max(values))
        return torch.sqrt(torch.as_tensor(sum(values) / len(values), device=eigenvalues.device))

    def estimate_noise_level_pca(self, batch):
        batch_size = batch.shape[0]
        noise_levels = torch.zeros(batch_size, device=batch.device)

        for i in range(batch_size):
            magnitude_image = self._magnitude(batch[i])
            patches = self.extract_patches(magnitude_image)
            eigenvalues = self.pca_torch(patches)
            noise_levels[i] = self.estimate_noise_level_from_eigenvalues(eigenvalues)

        return noise_levels

    def estimate_noise_level_mri_wavelet(self, batch, wavelet="db1", level=1):
        batch_size = batch.shape[0]
        noise_levels = torch.zeros(batch_size, device=batch.device)

        for i in range(batch_size):
            magnitude_image = self._magnitude(batch[i])
            noise_levels[i] = self.estimate_noise_level_wavelet(magnitude_image, wavelet, level)

        return noise_levels

    def estimate_noise_level_tv(self, batch):
        batch_size = batch.shape[0]
        noise_levels = torch.zeros(batch_size, device=batch.device)

        for i in range(batch_size):
            magnitude_image = self._magnitude(batch[i])
            image_np = magnitude_image.detach().cpu().numpy()
            denoised = denoise_tv_chambolle(image_np, weight=0.1)
            noise_levels[i] = np.sqrt(mean_squared_error(image_np, denoised))

        return noise_levels

    def estimate_noise_level_block_adaptive_gaussian(
        self,
        batch,
        block_size=16,
        kernel_size=5,
        sigma=1.0,
    ):
        batch_size = batch.shape[0]
        noise_levels = torch.zeros(batch_size, device=batch.device)

        for i in range(batch_size):
            magnitude_image = self._magnitude(batch[i])
            blocks = magnitude_image.unfold(0, block_size, block_size).unfold(1, block_size, block_size)
            blocks = blocks.contiguous().view(-1, block_size, block_size)
            block_stds = blocks.view(blocks.size(0), -1).std(dim=1)
            min_std = block_stds.min()
            smooth_blocks = blocks[block_stds <= min_std * 1.2]

            if smooth_blocks.size(0) == 0:
                noise_levels[i] = min_std
                continue

            smooth_blocks_filtered = smooth_blocks.clone()
            for j in range(smooth_blocks_filtered.size(0)):
                smooth_blocks_filtered[j] = apply_gaussian_blur(
                    smooth_blocks_filtered[j].unsqueeze(0),
                    kernel_size,
                    sigma,
                ).squeeze(0)

            diff_image = smooth_blocks - smooth_blocks_filtered
            noise_levels[i] = diff_image.view(diff_image.size(0), -1).std()

        return noise_levels

    def estimate_noise_level_adaptive_wavelet(self, batch, wavelet="db1"):
        batch_size = batch.shape[0]
        noise_levels = torch.zeros(batch_size, device=batch.device)

        for i in range(batch_size):
            magnitude_image = self._magnitude(batch[i])
            coeffs = pywt.wavedec2(magnitude_image.detach().cpu().numpy(), wavelet)
            finest_scale = torch.as_tensor(self._flatten_detail_coeffs(coeffs[-1]), device=batch.device)
            noise_levels[i] = self.mad(finest_scale)

        return noise_levels

    def estimate_noise_level_scale_invariance(self, batch, patch_size=None):
        patch_size = patch_size or self.patch_size

        def dct2d(x):
            x = x - x.mean(dim=(-2, -1), keepdim=True)
            result = torch.zeros_like(x)
            rows = torch.arange(patch_size, device=x.device)[:, None]
            cols = torch.arange(patch_size, device=x.device)[None, :]
            for u in range(patch_size):
                for v in range(patch_size):
                    basis = torch.cos(np.pi * (2 * rows + 1) * u / (2 * patch_size)) * torch.cos(
                        np.pi * (2 * cols + 1) * v / (2 * patch_size)
                    )
                    result[:, :, u, v] = (x * basis).sum(dim=(-2, -1)) * (2 / patch_size)
            return result

        def kurtosis(x):
            mean = np.mean(x)
            std = np.std(x)
            return np.mean((x - mean) ** 4) / (std**4 + 1e-8)

        def objective(params, kurt_values, var_values):
            kx, sigma2_n = params
            pred_kurt = (kx - 3) / (1 + sigma2_n / (var_values - sigma2_n + 1e-8)) ** 2 + 3
            return np.sum((pred_kurt - kurt_values) ** 2)

        batch_size = batch.shape[0]
        noise_levels = np.zeros(batch_size)

        for i in range(batch_size):
            magnitude_image = self._magnitude(batch[i])
            patches = F.unfold(
                magnitude_image.unsqueeze(0).unsqueeze(0),
                kernel_size=patch_size,
                stride=patch_size,
            )
            patches = patches.view(1, patch_size, patch_size, -1).permute(0, 3, 1, 2)
            dct_coeffs = dct2d(patches)
            dct_coeffs_flat = dct_coeffs[:, :, 1:, 1:].reshape(-1, (patch_size - 1) ** 2)
            dct_coeffs_np = dct_coeffs_flat.detach().cpu().numpy()

            var_values = np.var(dct_coeffs_np, axis=0)
            kurt_values = np.array([kurtosis(dct_coeffs_np[:, j]) for j in range(dct_coeffs_np.shape[1])])
            initial_guess = [np.mean(kurt_values), np.mean(var_values) / 10]
            result = minimize(
                objective,
                initial_guess,
                args=(kurt_values, var_values),
                bounds=[(0, None), (1e-6, None)],
            )
            noise_levels[i] = np.sqrt(result.x[1])

        return torch.as_tensor(noise_levels, device=batch.device, dtype=batch.dtype)

    def _magnitude(self, sample):
        return torch.abs(torch.complex(sample[0], sample[1]))

    def _flatten_detail_coeffs(self, detail_coeffs):
        return np.concatenate([detail_coeffs[0].ravel(), detail_coeffs[1].ravel(), detail_coeffs[2].ravel()])
