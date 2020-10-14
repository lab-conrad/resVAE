from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='resvae',
    version='1.0.0a',
    url='',
    author='Soeren Lukassen',
    description='Implementation of resVAE: restricted latent variational autoencoder',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3.0 or later (GPL-3.0-or-later)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'keras',
        'urllib3',
        'scikit-learn',
        'keras==2.3.1',
        'tensorflow-gpu==1.15.4',
        'numpy',
        'scipy',
        'h5py',
        'pandas',
        'matplotlib',
        'seaborn',
        'tables'
    ]
)
