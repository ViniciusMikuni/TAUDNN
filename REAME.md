# Low pT tau reconstruction DNN workflow

**Important** The setup is intended to use PSI T3 resources and is not tested elsewhere.

# Installation

To setup the required packages do 

```bash
source init.sh
source setup.sh
``` 

The first command takes a while to finish. The next time you only need to use:

```bash
source setup.sh
```

To properly set the environment variables.

# Analyzing the data

To verify that everything was properly setup you can try to run the code on a single file by doing

```
python analysis_tau.py --chunk 1000 --maxchunk 2 /pnfs/psi.ch/cms/trivcat/store/user/ytakahas/forVini/DNN_v4/2017/0000/flatTuple_124.root
```

If the test was succesfull, an ```.h5``` file should have been created in the ```\h5``` directory.

This file is the input required to train the classifier. Two different classifiers are provided: a multiclassifier that gives a score per particle collision and a part segmentation training, where each particle receives a score. 
 The file ```train_slurm.sh``` has a few examples for the training per class (```train_class.py```) and segmentation (```train.py```)
