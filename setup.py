from setuptools import setup, find_packages

setup(
    name="centigrad",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "triton",
        "numpy",
        "cupy-cuda12x",  # Adjust version based on your CUDA version
    ],
)
