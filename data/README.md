# Dataset

The dataset used in this research was generated synthetically using the parameters defined in `src/config.py`.

## Option 1: Download Pre-Generated Dataset (Recommended)

To save time, you can download the complete, pre-generated dataset used for our paper.

The dataset is hosted permanently on Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16260992.svg)](https://doi.org/10.5281/zenodo.16260992)


#### Instructions:
1.  Download the two `.zip` files from the link above.
2.  Unzip the contents.
3.  You should now have two folders: `graph_features` and `graph_objects`.
4.  Place both of these folders directly inside this `data/` directory.

## Option 2: Regenerate Dataset from Scratch

For full reproducibility, you can regenerate the entire dataset by running the following script from the project's root directory:
```bash
python scripts/generate_data.py
