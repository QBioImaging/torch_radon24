from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="torch_radon24",
    version="1.3",
    description="Radon Transformation for Pytorch 2.0 package",
    author="Minh Nhat Trinh",
    license="GNU GENERAL PUBLIC LICENSE",
    packages=["torch_radon24"],
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.25.0",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
