#!/bin/bash

case $CLUSTER in
        "anvil.rcac.purdue.edu")
		module load anaconda
		conda init bash
		export CONDA_PKGS_DIRS=$PROJECT/.conda/packages/
		export PIP_CACHE_DIR=$PROJECT/.pip/cache
		conda create --prefix $PROJECT/envs/workshop python=3.11 pip -y
		conda activate $PROJECT/envs/workshop
                ;;
        "ib.bridges2.psc.edu")

                ;;
        "expanse.sdsc.edu")

                ;;
        *)
                echo "ALERT: Cluster not found"
                ;;
esac
