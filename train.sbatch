#!/bin/bash
#
#SBATCH --job-name=mbodijob
#SBATCH --output=zzjob.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mail-type=END
#SBATCH --mail-user=whk240@nyu.edu

module purge;
module load anaconda3/2024.02;
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh;
conda activate /scratch/whk240/mbodi/;
export PATH=/scratch/whk240/mbodi/bin:$PATH;
export PYTHONPATH=/scratch/whk240/llm/torch_legacy:$PYTHONPATH;
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
cd /scratch/whk240/llm;
bash /scratch/whk240/llm/torch_legacy/train.sh