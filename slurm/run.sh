#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=NCA_diameter_128-hid_64-layer_lr-1e-04_10000-data_cutCorners_4
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sam.earle@nyu.edu
#SBATCH --output=/scratch/se2161/pathfinding-nca/runs/NCA_diameter_128-hid_64-layer_lr-1e-04_10000-data_cutCorners_4.out
#SBATCH --account=pr_174_tandon_advanced

export TUNE_RESULT_DIR='./ray_results/'

cd /scratch/se2161/pathfinding-nca

python -c "from main import *; main_experiment_load_cfg('/scratch/se2161/pathfinding-nca/slurm/auto_configs/NCA_diameter_128-hid_64-layer_lr-1e-04_10000-data_cutCorners_4.yaml')"

# leave trailing line
