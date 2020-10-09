from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='resvae',
    version='1.0.0a',
    url='',
    author='Soeren Lukassen',
    description='Implementation of a conditional variational autoencoder',
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
        'numpy',
        'pandas',
        'urllib3',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tables'
    ]
)