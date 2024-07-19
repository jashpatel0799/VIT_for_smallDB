#!/bin/bash
#SBATCH --job-name=mtp_adv_ddpm 	# Job name
#SBATCH --partition=dgx 	#Partition name can be test/small/medium/large/gpu #Partition “gpu” should be used only for gpu jobs
#SBATCH --nodes=1 			# Run all processes on a single node
#SBATCH --ntasks=1 			# Run a single task
#SBATCH --cpus-per-task=4 	# Number of CPU cores per task
#SBATCH --gres=gpu 		    # Include gpu for the task (only for GPU jobs)
##SBATCH --mem=8gb 			# Total memory limit
##SBATCH --time=10:00:00 	# Time limit hrs:min:sec
#SBATCH --output=./results/test_%j.log # Standard output and error log

## module load python/3.8

## pip3 -q install -U torchvision --user
# nvidia-smi


#echo "Training VIT"

# python3.8 -u train_adv.py
python3 -u main.py
##python3 -u main_q2.py
## python3 -u main_q2_modfied_arch.py
