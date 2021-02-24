# equivariant_image_recon 

[![DOI](https://zenodo.org/badge/341419972.svg)](https://zenodo.org/badge/latestdoi/341419972)

This package implements the methods described in ["Equivariant neural networks for inverse problems"](https://arxiv.org/abs/2102.11504).

## Installation

The package can be installed by running

```
git clone https://github.com/fsherry/equivariant_image_recon.git && cd equivariant_image_recon && pip install -e . 
```

## Usage

The scripts folder contains the scripts that were used to run the experiments in the paper, which can serve as an example of how to use the methods.
The shell scripts can be run with Slurm, or without if SLURM_ARRAY_TASK_ID is set appropriately.
The environment variable data_path should point to the filtered datasets generated with the jupyter notebooks in the notebooks folder and the environment variable save_path should point to the location where you want results to be saved.
The experiments use the [LIDC-IDRI dataset](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) for the CT reconstruction problems and the brain_fastMRI_DICOM subset of the [FastMRI dataset](https://fastmri.med.nyu.edu) and for the MRI reconstruction problems.

## License

This project is licensed under the GPLv3 license (see LICENSE.txt).
