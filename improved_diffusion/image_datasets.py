import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
import pathlib
import random
import h5py
import numpy as np
import os
import nibabel as nib
import meddlr.ops as oF
from meddlr.data.data_utils import collect_mask
from pathlib import Path
from meddlr.data.transforms.subsample import PoissonDiskMaskFunc, RandomMaskFunc1D
from meddlr.forward import SenseModel
import meddlr.ops.complex as cplx
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
class ScanData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, data_path, split, dim, echo, mask_func, sample_rate, seed):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        self.mask_func = mask_func
        self.seed = seed
        self.dim = dim
        self.echo = echo
        self.split = split
        self.accelerations = mask_func.accelerations
        root = Path(data_path) / Path('files_recon_calib-24_split') / Path(split)
        root_dicom = Path(data_path) / Path('image_files_split') / Path(split)
        root_seg = Path(data_path) / Path('segmentation_masks_split') / Path('raw-data-track') / Path(split)
        files = [item.split('.', 1)[0] for item in os.listdir(str(root))]

        if sample_rate < 1:
            # random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        self.files = files
        self.scans = []

        for fname in sorted(files):
            fname_recon = str(root) + '/' + fname + '.h5'
            fname_seg = str(root_seg) + '/' + fname + '.nii.gz'
            fname_dicom = str(root_dicom) + '/' + fname + '.h5'
            self.scans += [(fname, fname_recon, fname_seg, fname_dicom)]

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, i):
        echo = self.echo
        fname, fname_recon, fname_seg, fname_dicom = self.scans[i]
        kspace = h5py.File(fname_recon, 'r')['kspace'][:, :, :, echo, :]
        target_all = h5py.File(fname_recon, 'r')['target']
        target_all_abs = np.abs(target_all)
        target_std = np.std(target_all_abs)
        target_mean = np.mean(target_all_abs)
        target = target_all[:, :, :, echo, :]
        maps = h5py.File(fname_recon, 'r')['maps'][()]
        seg_mask = nib.load(fname_seg).dataobj[()]
        seg_mask = oF.categorical_to_one_hot(seg_mask, channel_dim=-1, background=0, num_categories=6)
        seg_mask = collect_mask(seg_mask, (0, 1, (2, 3), (4, 5)), out_channel_first=True)
        kspace = torch.as_tensor(kspace).unsqueeze(0)
        dim = 1
        kspace_kxkykz = oF.fftshift(torch.fft.fftn(oF.ifftshift(kspace, dim=dim), dim=dim, norm='ortho'), dim=dim)
        dim = 3
        kspace_kxkyz = oF.ifftshift(torch.fft.ifftn(oF.fftshift(kspace_kxkykz, dim=dim), dim=dim, norm='ortho'), dim=dim).squeeze(0)
        if "Random" in self.mask_func.__class__.__name__:
            from fastmri.data import transforms as T
            shape = (1, 512, 416, 1)
            mask, num_low_frequencies = self.mask_func(shape, None, seed=self.seed)
            mask = oF.zero_pad(mask, [1,512]).unsqueeze(-1)
            kspace_masked = kspace_kxkyz.unsqueeze(0) * mask
            kspace_bmask = 'random'
        else:
            mask = self.mask_func((1, 512, 416, 1), seed=self.seed, acceleration=None)
            kspace_masked, kspace_bmask = self.apply_mask(torch.as_tensor(kspace_kxkyz).unsqueeze(0), torch.as_tensor(mask))
        image_sense = self.sense_recon(kspace_masked, torch.as_tensor(maps).unsqueeze(0), kspace_bmask)

        data = {'fname': fname ,'kspace_masked': kspace_masked.squeeze(0).numpy(), 'kspace_kxkyz': kspace_kxkyz, 'target': target,
                'maps': maps, 'seg_mask': seg_mask, 'image_sense': image_sense, #'image_sense_full': image_sense_full, #'image_zf': image_zf,
                'target_mean': target_mean, 'target_std': target_std} #, 'dicom_echo1': dicom_echo1, 'dicom_seg_mask': dicom_seg_mask}

        return data


    def get_specific_item(self, i):
        return self.__getitem__(i)

    def apply_mask(self, kspace, mask):
        mask = oF.zero_pad(mask, kspace.shape[1:3]).unsqueeze(-1)
        return torch.where(mask == 0, torch.tensor([0], dtype=kspace.dtype), kspace), mask.squeeze(-1)

    def zero_filled(self, kspace_masked):
        image_zf = torch.zeros((kspace_masked.shape[0], kspace_masked.shape[1], kspace_masked.shape[2], kspace_masked.shape[3], 1), dtype=torch.complex64)
        for slice in range(kspace_masked.shape[3]):
            image_zf[:, :, :, slice, ...] = torch.sum(oF.ifft2c(kspace_masked[:, :, :, slice, ...], channels_last=True), dim=-1).unsqueeze(-1)
        return image_zf.squeeze(0).numpy()

    def sense_recon(self, kspace_masked, maps, kspace_bmask):
        bmask = kspace_bmask
        image_sense = torch.zeros((maps.shape[0], maps.shape[1], maps.shape[2], maps.shape[3], 1), dtype=torch.complex64)
        for slice in range(kspace_masked.shape[3]):
            if bmask == 'random':
                kspace_bmask = cplx.get_mask(kspace_masked[:, :, :, slice, ...])
            A = SenseModel(maps[:, :, :, slice, ...], weights=kspace_bmask)
            image_sense[:, :, :, slice, ...] = A(kspace_masked[:, :, :, slice, ...], adjoint=True)
        return image_sense.squeeze(0).numpy()


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, data_path, split, dim, echo, mask_func, sample_rate, seed):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        self.mask_func = mask_func
        self.seed = seed
        self.dim = dim
        self.echo = echo
        npy_path = data_path / Path('dim_' + str(dim)) / Path('echo_' + str(echo)) / Path(split)
        files = list(Path(npy_path).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        self.files = files
        self.rng = np.random.RandomState(seed=seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file_path = self.files[i]
        dict = np.load(file_path, allow_pickle=True)
        kspace, maps, target, seg_mask = dict.item().get('kspace'), dict.item().get('maps'), dict.item().get('target'), dict.item().get('seg_mask')
        target_mean, target_std = dict.item().get('target_mean'), dict.item().get('target_std')
        kspace_masked, kspace_usmask = self.apply_mask(torch.as_tensor(kspace).unsqueeze(0))
        # image_sense = self.sense_recon(kspace_masked, torch.as_tensor(maps).unsqueeze(0), kspace_usmask)
        image_zf = self.zero_filled(kspace_masked)
        data = {'fname': file_path.stem,
                'kspace_full': kspace, 'kspace_masked': kspace_masked.squeeze(0).numpy(), 'target': target, 'image_zf': image_zf,
                'maps': maps, 'seg_mask': seg_mask, 'kspace_usmask': kspace_usmask,
                'target_mean': target_mean, 'target_std': target_std}
        return data

    def apply_mask(self, kspace):
        """
        Apply the dataset's mask_func instead of relying on precomputed masks on disk.
        Supports both RandomMaskFunc (fastMRI) and PoissonDiskMaskFunc (meddlr).
        """
        shape = (1, kspace.shape[1], kspace.shape[2], 1)
        if "Random" in self.mask_func.__class__.__name__:
            mask, _ = self.mask_func(shape, None, seed=self.rng.randint(0, 1 << 16))
        else:
            mask = self.mask_func(shape, seed=self.rng.randint(0, 1 << 16), acceleration=None)

        # Ensure mask matches spatial dims, then broadcast over coils
        if mask.dim() == 3:  # (1,H,W)
            mask = mask.unsqueeze(-1)
        mask = oF.zero_pad(mask, kspace.shape[1:3])
        kspace_masked = torch.where(
            mask == 0,
            torch.tensor([0], dtype=kspace.dtype, device=kspace.device),
            kspace,
        )
        return kspace_masked, mask.squeeze(-1)

    def zero_filled(self, kspace_masked):
        image_zf = torch.sum(oF.ifft2c(kspace_masked, channels_last=True), dim=-1)
        return image_zf.squeeze(0).unsqueeze(-1).numpy()



def create_mask_func(args):
    if args.mask_pattern == 'random':
        from fastmri.data.subsample import RandomMaskFunc
        mask_func = RandomMaskFunc(center_fractions=[0.08], accelerations=[args.acc_rate])
    elif args.mask_pattern == 'poisson':
        mask_func = PoissonDiskMaskFunc(accelerations=(args.acc_rate,), calib_size=24, center_fractions=(), crop_corner=True, max_attempts=5, module='sigpy')
    else:
        raise ValueError(f"unsupported mask pattern: {args.mask_pattern}")
    return mask_func


def create_datasets(args):
    mask_func = create_mask_func(args)

    data_train = SliceData(
        data_path=args.data_path_npy,
        split='train',
        dim=args.slice_dim,
        echo=args.echo,
        mask_func=mask_func,
        sample_rate=args.sample_rate_train,
        seed=args.seed
    )

    data_val = ScanData(
        data_path=args.data_path,
        split='val',
        dim=args.slice_dim,
        echo=args.echo,
        mask_func=mask_func,
        sample_rate=args.sample_rate_test,
        seed=args.seed
    )

    data_test = ScanData(
        data_path=args.data_path,
        split='test',
        dim=args.slice_dim,
        echo=args.echo,
        mask_func=mask_func,
        sample_rate=args.sample_rate_test,
        seed=args.seed
    )

    return data_train, data_val, data_test

def create_data_loaders(args):
    data_train, data_val, data_test = create_datasets(args)
    train_sampler = DistributedSampler(dataset=data_train, shuffle=True, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    # batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    # loader_train = DataLoader(data_train, batch_sampler=batch_sampler_train, num_workers=args.data_train_num_worker,
    #                           pin_memory=True)
    loader_train = DataLoader(data_train, batch_size=args.batch_size, sampler=train_sampler, pin_memory=True, num_workers=args.data_train_num_worker)

    # loader_train = DataLoader(
    #     dataset=data_train,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.data_train_num_worker,
    #     pin_memory=True
    # )

    # loader_val = DataLoader(
    #     dataset=data_val,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=args.data_eval_num_worker
    # )

    # loader_test = DataLoader(
    #     dataset=data_test,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=args.data_eval_num_worker,
    # )

    while True:
        yield from loader_train

def create_val_data_loaders(args):
    data_train, data_val, data_test = create_datasets(args)

    # loader_train = DataLoader(
    #     dataset=data_train,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.data_train_num_worker,
    #     pin_memory=True
    # )
    val_sampler = DistributedSampler(dataset=data_val, shuffle=False, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    loader_val = DataLoader(data_val, batch_size=1, sampler=val_sampler, num_workers=args.data_eval_num_worker)

    # loader_val = DataLoader(
    #     dataset=data_val,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=args.data_eval_num_worker
    # )

    # loader_test = DataLoader(
    #     dataset=data_test,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=args.data_eval_num_worker,
    # )

    return loader_val

def create_test_data_loaders(args):
    mask_func = create_mask_func(args)
    data_test = ScanData(
        data_path=args.data_path,
        split='test',
        dim=args.slice_dim,
        echo=args.echo,
        mask_func=mask_func,
        sample_rate=args.sample_rate_test,
        seed=args.seed
    )

    # loader_train = DataLoader(
    #     dataset=data_train,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.data_train_num_worker,
    #     pin_memory=True
    # )
    # test_sampler = DistributedSampler(dataset=data_test, shuffle=False, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    # loader_test = DataLoader(data_test, batch_size=1, sampler=test_sampler, num_workers=args.data_eval_num_worker)

    # loader_val = DataLoader(
    #     dataset=data_val,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=args.data_eval_num_worker
    # )

    loader_test = DataLoader(
        dataset=data_test,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=args.data_eval_num_worker,
    )

    return loader_test

def create_visual_test_data_loaders(args):
    data_train, data_val, data_test = create_datasets(args)

    test_sampler = DistributedSampler(dataset=data_test, shuffle=False, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    loader_val = DataLoader(data_test, batch_size=1, sampler=test_sampler)

    # loader_val = DataLoader(
    #     dataset=data_val,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=args.data_eval_num_worker
    # )

    # loader_test = DataLoader(
    #     dataset=data_test,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=args.data_eval_num_worker,
    # )

    return data_test
