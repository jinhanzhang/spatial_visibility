#!/bin/bash

#SBATCH --account=pr_31_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=128GB	
#SBATCH --gres=gpu:1
#SBATCH --job-name=graphgru
#SBATCH --output=./log/multi_mse_lr1e4_G1_vs128_fov_dim128_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jz5952@nyu.edu

module purge

singularity exec --nv \
	--overlay /scratch/$USER/spatial-vis/overlay-25GB-500K.ext3:ro \
		/scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
		/bin/bash -c "source /ext3/env.sh && conda activate py310; python3 graphgru.py"