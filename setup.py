from setuptools import find_packages, setup


setup(
    name="nad-mri",
    packages=find_packages(),
    install_requires=[
        "blobfile>=1.0.5",
        "fastmri",
        "h5py",
        "matplotlib",
        "meddlr",
        "mpi4py",
        "nibabel",
        "numpy",
        "PyWavelets",
        "scikit-image",
        "scipy",
        "sigpy",
        "torch",
        "tqdm",
    ],
)
