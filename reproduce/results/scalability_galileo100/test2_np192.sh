#!/bin/bash
#SBATCH --job-name=test2_np192
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=48
#SBATCH --time=00:30:00
#SBATCH --output=/g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/scalability_galileo100/test2_np192.log
#SBATCH --account=uBS26_MatGer
#SBATCH --partition=g100_usr_prod
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
module load profile/global
module load intel/oneapi-2021--binary intelmpi/oneapi-2021--binary mkl/oneapi-2021--binary
module load gnu/10.2.0--gcc--8.3.1
module load libszip/2.1.1--gcc--10.2.0 zlib/1.2.11--gcc--10.2.0
module load boost/1.76.0--intelmpi--oneapi-2021--binary
module load hdf5/1.10.7--intelmpi--oneapi-2021--binary
module load python/3.8.6--gcc--10.2.0
module load cmake
srun /g100_work/uBS26_MatGer/dealii-internodes/build/examples/coupled_diffusion/coupled_diffusion /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/scalability_galileo100/test2_np192.prm
