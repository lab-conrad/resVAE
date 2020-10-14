# resVAE - a restricted latent variational autoencoder

[![DOI](https://zenodo.org/badge/302657137.svg)](https://zenodo.org/badge/latestdoi/302657137)

[Preprint on BioRxiv](https://www.biorxiv.org/content/10.1101/740415v2)

resVAE is a restricted latent variational autoencoder that we wrote to uncover hidden structures in gene expression data, especially using single-cell RNA sequencing. In principle it can be used with any hierarchically structured data though, so feel free to play around with it.

## How does resVAE work?

Briefly, resVAE is not too different from a standard variational autoencoder. In case you are not familiar with artificial neural networks, imagine an algorithm that compresses data, forces the compressed representation to have a specific distribution (in our case, a Gaussian), and decompresses the data again. If you are more familiar with (variational) autoencoders, but fancy a quick reminder, have a look at this excellent [explanation](https://arxiv.org/abs/1906.02691) by Kingma and Welling.
resVAE deviates from this idea in some aspects. One is that we use preclustered data, and feed the identity function of those clusters to the latent space of our network. Thus, we reserve dimensions for individual classes such as cell types, while keeping the encoder and decoder parts of the network the same. In the context of gene expression, this forces the network to learn features that are shared across cell types, but may be more or less active in one cell type or the other. Having reserved dimensions in the latent space allows us to easily map these features to say cell types or disease states. As it turns out, this enables the identification of functional gene sets, including the possibility of correcting this gene set inference for batch effects or treatment groups by encoding these in the latent variable space.

For more information regarding resVAE, please read our [preprint on BioRxiv](https://www.biorxiv.org/content/10.1101/740415v2).

## Getting started

### Prerequisites and installation

Although we tried to keep the list of dependencies short, resVAE does require you to download some Python packages. If you are using the install script we provide, these should be handled automatically. This is as easy as downloading the project using `git clone` on the project link (or pressing the button on the top right of the main project page), opening the downloaded folder using a command line, and running:
```
pip install -e .
```
In case of any issues, you can try forcing pip to install any dependencies first:
```
pip install -r requirements.txt
```
If this still does not work, try installing the failing packages individually using `conda install <package>` or `pip install <package>`.

You can also run resVAE from the project directory without installing it, the options in the example notebook should be fully compatible with this.

**⚠️ Please note that you would need `tensorflow or tensorflow-gpu version 1` and `keras version < 2.4` to run the current version of resVAE.**

### Required input files

To run resVAE, you will need at the very minimum a single-cell gene expression matrix or something similar where you have a format of *samples x features* or *features x samples*. This should be in some delimited text format, with row and column names included. Ideally, you will also have a file with cluster identities. For a somewhat more complex use case, this could also be several different classes, such as cell type, treatment, disease status, and batch. Each of these variables can then be converted to a one-hot format (resVAE contains utility functions to do this) and concatenated.

### Running resVAE

An example project demonstrating the workflow of resVAE is [included](Example_notebook.ipynb). A more complete documentation of the API (although this is still very much work in progress) can be found in the `docs/_build/html/` subfolder.

## Contributing

We welcome any contributions to the project, but ask you to adhere to some rules laid out in [CONTRIBUTING.md](CONTRIBUTING.md).

## Licensing

This work is released under the terms of the [GNU GPLv3 license](LICENSE.md). Note that this is a copyleft license, so by using resVAE in your own projects, you agree to license them under a compatible license. In case you are interested in including resVAE or parts of its code in your own projects but cannot comply with this, please contact us directly.

