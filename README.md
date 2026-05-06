# A Stability Benchmark of Generative Regularizers for Inverse Problems

> Generative (diffusion) priors demonstrate remarkable performance in addressing inverse problems in imaging. Yet, for scientific and medical imaging, it is crucial that reconstruction techniques remain stable and reliable under imperfect settings. Typical definitions of stability encompass the notion of ``convergent regularization'', robustness to out-of-distribution data, and to inaccuracies in the forward operator or noise model. We evaluate these properties numerically. Furthermore, we benchmark generative approaches against modern optimization-based methods inspired by the widely used variational techniques. Our results give insights for which settings and applications generative priors can deliver state-of-the-art reconstructions, and on those in which they fall short or may even be problematic.

We want to compare different generative and non-generative algorithms for image reconstruction and image restoration. 

## Preparation

Download the pretrained models and the validation and test datasets by

```
bash download_models_and_data.sh
```


#### Environment setup

Install Conda (Miniconda or Anaconda).

Create the environment from the file:
```
conda env create -f environment.yml
```

We noticed that some Linux systems have an outdated libstdc++ that breaks Diffusers binaries (error version `CXXABI_1.3.15' not found). In this case, setting LD_LIBRARY_PATH fixes it:
```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### Model Retraining

We provide the pretrained models (as described above) with which we generated the results. Alternatively, we also provide our training code for the diffusion and flow matching models. To this end, first the training datasets have to be downloaded as follows:
- [Walnut](https://zenodo.org/records/2686726): The dataset is hosted on zenodo. The files have to be downloaded to **dataset/walnut**.
- [AAPM](https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h): The AAPM dataset is publicly available. We use the B30 reconstruction kernel with full-dose 1 mm slice thickness. The files have to be downloaded to **dataset/aapm**. Preprocessing is done using **dataset/preprocess_aapm.py**.
- [CelebAHQ](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256): The CelebA-HQ dataset is availble on kaggle. 
- Ellipses: The ellipses are generated on the fly using **dataset/generate_ellipses.py**.

After downloading the datasets, run for the diffusion models
```
python diffusers_train.py --dataset_name walnut
```
(where `walnut` can also be replaced by `aapm` or `ellipses`).
For the flow matching models run
```
PYTHONPATH="$PYTHONPATH:./" python flow_matching_training/train_fm.py --dataset celebahq
```
(where `celebahq` can be replaced by `aapm`, `walnut` and `ellipses`).

## Reproduction of the results

The results from the paper can be reproduced by the python scripts 
- `main.py` (diffusion-based methods for CT `DiffPIR`, `DPS`, `DMPlug` and `REDdiff`), 
- `main_natural_images.py` (diffusion-based methods for natural images `DiffPIR`, `DPS`, `DMPlug` and `REDdiff`), 
- `main_learned_reg.py` (learned regularizers `WCRR`, `LSR` and `PnP-LSR`), 
- `main_RAM.py` (`RAM` for natural images)
- `main_tv.py` (total variation `TV`)
- `main_pnpflow.py` (`PnP-flow`)
- `main_flow.py` (text-to-image diffusion `FlowDPS`)

For calling these methods with the exact hyperparamters used for the paper, we provide shell scripts in the subdirectory `script_test`.
To reproduce all CT results for `DiffPIR` run
```
bash script_test/run_test_ct_diffpir.sh
```
For the other methods replace `diffpir` by `dmplug`, `dps`, `flowdps`, `lsr`, `pnpflow`, `pnplsr`, `reddiff`, `tv` or `wcrr`.
For the results for natural images replace `ct` by `natural` in the above command.

The results will be saved in the directory `results` by the subdirectory `{TRAINING_DATASET}_to_{TESTDATASET}/task_{TASK}/sigma_n={NOISE_LEVEL}/{METHOD}/test` which contains a text file summarizing the metrics and a subdirectory containing the images and a json file with the per-image metrics.