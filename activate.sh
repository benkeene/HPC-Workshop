#!/bin/bash

CLUSTER=$(hostname -d)
case $CLUSTER in
	"anvil.rcac.purdue.edu")
		conda activate /anvil/projects/x-cis240473/envs/workshop
		;;
	"ib.bridges2.psc.edu")
		conda activate /ocean/projects/cis240124p/$(whoami)/envs/workshop
		;;
	"expanse.sdsc.edu")
		conda activate /expanse/lustre/projects/ucf122/$(whoami)/envs/workshop
		;;
	*)
		echo "ALERT: Cluster not found"
		;;
