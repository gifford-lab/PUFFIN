# PUFFIN

## Download the trained models

Here we provide PUFFIN models trained on peptitde-MHC binding affinity datasets from [NetMHCpan3.0](http://www.cbs.dtu.dk/services/NetMHCpan-3.0/) (for class I MHC) and [NetMHCIIpan-3.2](http://www.cbs.dtu.dk/services/NetMHCIIpan-3.2/) (for class II MHC). The five folds are combined and split into training and validation set.

Note that because here we no longer need to hold out data as test set, this training setup is different from what's used in the paper (Table 1 and Table 2), where the training/test set provided by Bhattacharya et al. was used for class I MHC and the 5-fold cross-validation split from [NetMHCIIpan-3.2](http://www.cbs.dtu.dk/services/NetMHCIIpan-3.2/) was used for class II MHC (the performance on each fold was evaluated by a model trained on the other four folds).

Download the trained model by running the following in the repository folder:

```
wget http://gerv.csail.mit.edu/models.tar.gz
tar -zxvf models.tar.gz
```

## Set up the environment

We provide a [Conda](https://docs.conda.io/en/latest/) environment that provides all the Python packages required by PUFFIN. Build and activate this environment by:

```
conda env create -f environment.yml
source activate puffin
```

To deactivate this environment:

```
source deactivate
```


## Preprocess the input MHC-peptide pairs

Save all MHC-peptide pairs to evaluate in a tab-delimited file with three columns, each of which denotes the peptide sequence, the observed binding affinity (use any placeholder number when it is not available), and the MHC allele respectively. The MHC allele names supported are in the first column of [this](https://github.com/gifford-lab/PUFFIN/blob/master/data/MHC_pseudo.dat) file. ([class I example](https://github.com/gifford-lab/PUFFIN/blob/master/examples/toydata_class1), [class II example](https://github.com/gifford-lab/PUFFIN/blob/master/examples/toydata_class2))

Then preprocess the data by:

```
python preprocess.py -i DATAFILE -o OUTDIR -c CLASS
```

- `DATAFILE`: the file that contains the MHC-peptide pairs
- `OUTDIR`: the directory to save all the output
- `CLASS`: "1" for class I and "2" for class II


## Make predictions using the trained models

```
python score.py -o OUTDIR -c CLASS -g GPU
```

- `OUTDIR`: same as above
- `CLASS`: same as above
- `GPU`: a comma-delimited string that denotes the index(es) of the GPU(s) to run the models on (eg. "0,1,2,3"). We recommend using multiple GPUs if possible to speed up the prediction.

The predictions are saved in `$OUTDIR/PUFFIN.combined`. It's a tab-delimited file with four columns, each of which denotes the predicted mean affinity, epistemic uncertainty, aleatoric uncertainty, and the binding likelihood for a 500 nM binding threshold.


