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
                ;;
        "ib.bridges2.psc.edu")
		module load anaconda3
		conda init bash
		export CONDA_PKGS_DIR=/ocean/projects/cis240124p/$(whoami)/.conda/packages
		export PIP_CACHE_DIR=/ocean/projects/cis240124p/$(whoami)/.conda/packages
		conda create --prefix /ocean/projects/cis240124p/envs/workshop python=3.11 pip -y
		conda activate /ocean/projects/cis240124p/envs/workshop
                ;;
        "expanse.sdsc.edu")

                ;;
        *)
                echo "ALERT: Cluster not found"
                ;;
esac
