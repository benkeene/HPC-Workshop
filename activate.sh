#!/bin/bash

CLUSTER=$(hostname -d)
case $CLUSTER in
"anvil.rcac.purdue.edu")
	module load anaconda
	conda activate /anvil/projects/x-cis240473/envs/workshop
	;;
"ib.bridges2.psc.edu")
	module load anaconda3
	conda activate /ocean/projects/cis240124p/$(whoami)/envs/workshop
	;;
"expanse.sdsc.edu")
	module purge
	module load slurm sdsc cpu/0.15.4 gcc/10.2.0 anaconda3
	conda activate /expanse/lustre/projects/ucf122/$(whoami)/envs/workshop
	;;
*)
	echo "ALERT: Cluster not found"
	;;
esac
