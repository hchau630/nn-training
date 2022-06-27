from setuptools import setup, find_packages

setup(
    name='nn_training',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'torch',
        'torchvision',
    ]
)