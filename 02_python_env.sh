#!/bin/bash

CLUSTER=$(hostname -d)
case $CLUSTER in
        "anvil.rcac.purdue.edu")
		module load anaconda
		conda init bash
		export CONDA_PKGS_DIRS=/anvil/projects/x-cis240473/.conda/packages/
		export PIP_CACHE_DIR=/anvil/projects/x-cis240473/.pip/cache
		conda create --prefix /anvil/projects/x-cis240473/envs/workshop python=3.11 pip -y
		conda activate /anvil/projects/x-cis240473/envs/workshop
		pip install ipykernel
		python -m ipykernel install --prefix=/anvil/projects/x-cis240473/envs/workshop --display-name="Python (workshop)"
                ;;
        "ib.bridges2.psc.edu")
		module load anaconda3
		conda init bash
		export CONDA_PKGS_DIR=/ocean/projects/cis240124p/$(whoami)/.conda/packages
		export PIP_CACHE_DIR=/ocean/projects/cis240124p/$(whoami)/.pip/cache
		conda create --prefix /ocean/projects/cis240124p/$(whoami)/envs/workshop python=3.11 pip -y
		conda activate /ocean/projects/cis240124p/$(whoami)/envs/workshop
		pip install ipykernel
		python -m ipykernel install --prefix=/ocean/projects/cis240124p/$(whoami)/envs/workshop --display-name="Python (workshop)"
                ;;
        "expanse.sdsc.edu")
		module purge
		module load slurm sdsc cpu/0.15.4 gcc/10.2.0 anaconda3
		export CONDA_PKGS_DIR=/expanse/lustre/projects/ucf122/$(whoami)/.conda/packages/
		export PIP_CACHE_DIR=/expanse/lustre/projects/ucf122/$(whoami)/.pip/cache
		conda create --prefix /expanse/lustre/projects/ucf122/$(whoami)/envs/workshop python=3.11 pip -y
		conda activate /expanse/lustre/projects/ucf122/$(whoami)/envs/workshop
		pip install ipykernel
		python -m ipykernel install --prefix=/expanse/lustre/projects/ucf122/$(whoami)/envs/workshop --display-name="Python (workshop)"
                ;;
        *)
                echo "ALERT: Cluster not found"
                ;;
esac
