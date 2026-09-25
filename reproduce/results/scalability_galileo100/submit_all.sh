#!/bin/bash
cd /g100_work/uBS26_MatGer/dealii-internodes/reproduce
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/scalability_galileo100/test1_np48.sh
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/scalability_galileo100/test2_np48.sh
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/scalability_galileo100/test3_np48.sh
