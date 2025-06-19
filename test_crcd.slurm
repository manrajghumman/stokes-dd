#!/usr/bin/env bash

## ------------------------------------------------------------------
## Slurm directives defining the resource request 
## ------------------------------------------------------------------
#SBATCH --job-name=test_stokes_1
#SBATCH --output=test_stokes_1.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --time=1:00:00

## ---------------------------------------------------------------------
## Load software into environment 
## ---------------------------------------------------------------------
module purge
module load gcc/12.2.0   cmake/3.27.7   openmpi/4.1.5
module load eigen/3.4.0	 dealii/9.6.0

## ---------------------------------------------------------------------
## Setup software execution environment
## ---------------------------------------------------------------------
# Define environmental variables for Amber input/output files
OUT=output

# Define software executable
Stokes=./build/StokesDD > output.txt

# Display environmental variables to Slurm output file for diagnostics
echo DEAL_II_DIR    $DEAL_II_DIR
echo SLURM_NTASKS $SLURM_NTASKS
echo which Stokes `which $Stokes`
echo "Running on node:" `hostname`

# Display NVIDIA GPU information to Slurm output file
nvidia-smi

# Software execution line
mpirun -n   $SLURM_NTASKS  $Stokes   > output.txt 2>&1