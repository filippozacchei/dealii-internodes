#!/bin/bash
cd /g100_work/uBS26_MatGer/dealii-internodes/reproduce
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/scalability_galileo100-smoke/test1_np48.sh
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/scalability_galileo100-smoke/test1_np96.sh
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/scalability_galileo100-smoke/test2_np48.sh
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/scalability_galileo100-smoke/test2_np96.sh
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/scalability_galileo100-smoke/test3_np48.sh
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/scalability_galileo100-smoke/test3_np96.sh
