from setuptools import setup, find_packages

setup(
    name="thesis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "pyyaml>=6.0",
    ],
    author="Sam Vermeulen",
    author_email="vermeulen.sam.j@example.com",
    description="Code for my thesis",
    keywords="deep-learning, pytorch, cuda",
    python_requires=">=3.8",
)
