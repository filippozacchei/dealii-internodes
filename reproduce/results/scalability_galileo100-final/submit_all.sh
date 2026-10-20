#!/bin/bash
cd /g100_work/uBS26_MatGer/dealii-internodes/reproduce
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/scalability_galileo100-final/test6_np48.sh
sbatch /g100_work/uBS26_MatGer/dealii-internodes/reproduce/results/scalability_galileo100-final/test6_np768.sh
