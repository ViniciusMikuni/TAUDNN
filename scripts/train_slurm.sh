#!/bin/bash
#
#SBATCH --job-name=train_tau
#SBATCH --account=gpu_gres               # to access gpu resources
#SBATCH --partition=gpu
#SBATCH --nodes=1                      # request to run job on single node
#SBATCH --ntasks=5                     # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)      
#SBATCH --gres=gpu:1                     # request  for two GPU's on machine, this is total  amount of GPUs for job        
#SBATCH --mem=20000                     # memory (per node)
#SBATCH --time=0-27:30                   # time  in format DD-HH:MM
#SBATCH -e slurm/slurm-gpu-%A.err
#SBATCH -o slurm/slurm-gpu-%A.out

# Slurm reserves two GPU's (according to requirement above), those ones that are recorded in shell variable CUDA_VISIBLE_DEVICES
echo CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES
echo "--------------------------------------------------------------------"
env
echo "--------------------------------------------------------------------"
#cd $COFFEAHOME/ABCNet/scripts
source activate TFGPU
pyenv versions
echo "--------------------------------------------------------------------"
nvidia-smi
echo "--------------------------------------------------------------------"



#python train_class.py --log_dir TAU_MCBKG --batch 64
python train_class.py --log_dir TAU_DATABKG_dev --batch 64
#python train_class.py --params [10,1,16,64,128,1,64,128,128,256,128,256] --log_dir TAU_class_v2 --batch 64
#python train_class.py --params [20,1,16,64,128,1,64,64,128,256,128,128] --log_dir TAU_class_v21 --batch 64
#python train_class.py --params [20,1,16,64,128,1,64,64,128,256,256,256] --log_dir TAU_class_v22 --batch 64


#python train.py --log_dir TAU_UL17 --year 17
#python train.py --log_dir TAU_UL18 --year 18
