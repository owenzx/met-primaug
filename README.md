# Mutual Exclusivity Training and Primitive Augmentation to Induce Compositionality
This repository contains the official code for the paper:

[Mutual Exclusivity Training and Primitive Augmentation to Induce Compositionality](https://arxiv.org/)

Yichen Jiang*, Xiang Zhou* and Mohit Bansal

EMNLP 2022

### Dependencies

The code is tested on Python 3.7.9 and PyTorch 1.8.1.

Other dependencies are listed in `requirements.txt` and can be installed by running `pip install -r requirements.txt`

This repository uses [wandb](https://github.com/wandb/client) for logging experiments. So before running your experiments, you need to log in you wandb account.


### Datasets and Preprocessing
The preprocessed datasets and resources used in our experiments can be downloaded at [here](https://drive.google.com/file/d/1q6Lq-M1u89_yy0CPm4jF_7EHjnfGiggZ/view?usp=sharing).

The instructions below are used to create our datasets:

1. Get the SCAN dataset from [here](https://github.com/brendenlake/SCAN/tree/9da9c8af84509a66e10dfbdfefb3d1e83f3d0bea) with the MCD splits. Get the COGS dataset from [here](https://github.com/najoungkim/COGS). Put the two folders under the root directory of this repository.
2. Prepare the validation dataset by run `python dev_generator.py --exp_type [exp_type]`. Here `[exp_type]` is the name of the dataset (or split in the SCAN dataset). Please refer to the code for details.
3. For MET related experiments, we need to pre-cluster the words in the datasets. The precomputed files can be downloaded along with the preprocessed dataset at [here](https://drive.google.com/file/d/1q6Lq-M1u89_yy0CPm4jF_7EHjnfGiggZ/view?usp=sharing). We have also included the script for reproducing the clustering on COGS at `substitutors/cogs_substitutor.py`.

#### prim2primX Data Augmentation
See the `prim2primX` folder for detailed instruction on how to run the data augmentation.
Once you have the augmented data, simply set:
* `--special_train_data=train_primx2` for SCAN addprim_jump;
* `--special_train_data=train_primx2s2` for COGS. 

### Experiments

We provide scripts to run experiments on both the SCAN and the COGS dataset. The corresponding scripts are under `scripts/SCAN` and `scripts/COGS`

Under each directory, we provide the following scripts:

* `transformer_baseline.sh`
* `LSTM_baseline.sh`
* `transformer_met.sh`
* `LSTM_met.sh`
* `transformer_prim2primx.sh`
* `LSTM_prim2primx.sh`
* `transformer_metprim.sh`
* `LSTM_metprim.sh`

The first part of the file name (`LSTM` or `transformer`) denotes the baseline architecture. 

The second part of the name denotes the method type. `baseline` stands for baselines, `met` stands for MET experiments. `prim2primx` stands for prim2primX data augmentation experiments, and `metprim` stands for the combination of MET and prim2primX.

Before running the experiments, you need to set the `home` variable and the `savefolder` variable in these scripts. Then, to run the experiments, simply execute the corresponding scripts `bash scripts/COGS/transformer_met.sh` 

#### Ablations
We also include scripts for several ablations in our paper. 
1. For the MET-meta variant of our model, an example script is provided at `COGS/transformer_metmeta.sh`. The same change can also be applied to other scripts to get this variant for other datasets. Additionally, the experiment in Table 11 can be reproduced by replacing `--meta_loss_type unlikelihood` with `--meta_loss_type unlikelihood2` (Avg-Word Unlike) and `--meta_loss_type unlikelihood3` (Min-Word Unlike) respectively.
2. For the synthetic experiment used in Appendix C.1, the scripts are in `scripts/toy`, and the data can be downloaded at [here](some link). 



### Acknowledgement
The code in this repository is based on [https://github.com/ekinakyurek/lexical](https://github.com/ekinakyurek/lexical)

### Reference
```
@inproceedings{jiang2022mutual,
  title={Mutual Exclusivity Training and Primitive Augmentation to Induce Compositionality},
  author={Yichen Jiang and Xiang Zhou and Mohit Bansal},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2022}
}
```
