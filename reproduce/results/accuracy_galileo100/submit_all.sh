#!/bin/bash
cd /g100_work/uBS26_MatGer/dealii-internodes/reproduce
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/accuracy_galileo100/gmres_pc_k2.sh
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/accuracy_galileo100/gmres_nopc_k2.sh
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/accuracy_galileo100/gmres_pc_k3.sh
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/accuracy_galileo100/gmres_nopc_k3.sh
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/accuracy_galileo100/gmres_pc_k4.sh
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/accuracy_galileo100/gmres_nopc_k4.sh
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/accuracy_galileo100/gmres_pc_k5.sh
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/accuracy_galileo100/gmres_nopc_k5.sh
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/accuracy_galileo100/gmres_pc_k6.sh
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/accuracy_galileo100/gmres_nopc_k6.sh
