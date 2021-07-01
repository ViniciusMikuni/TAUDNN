#!/usr/bin/env bash

source /cvmfs/sft.cern.ch/lcg/views/LCG_97python3/x86_64-centos7-gcc8-opt/setup.sh
pyenv virtualenv COFFEA
source activate COFFEA
python -m pip install --user pip --upgrade
python -m pip install --user  coffea[parsl]==0.6.44

if [ ! -d "$HOME/bin" ]
then
    echo "Creating bin folder under the home area."
    mkdir $HOME/bin
fi

export PARSLINSTALL=$(python -c "import parsl; print(parsl.__path__[0])")
ln -s $PARSLINSTALL/executors/high_throughput/process_worker_pool.py $HOME/bin
chmod +x $HOME/bin/process_worker_pool.py
