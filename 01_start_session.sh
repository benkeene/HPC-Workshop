#!/bin/bash

identify_cluster() {
	local domain=$(hostname -d)
	if [[ $domain == "anvil.rcac.purdue.edu" ]]; then
		echo "You're on Purdue Anvil"
	elif [[ $domain == "ib.bridges2.psc.edu" ]]; then
		echo "You're on PSC Bridges-2"
	elif [[ $domain == "expanse.sdsc.edu" ]]; then
		echo "You're on SDSC Expanse"
	else
		echo "ALERT: Cluster not found"
	fi
}

CLUSTER=$(identify_cluster)
echo $CLUSTER

case $CLUSTER in
	"anvil.rcac.purdue.edu")
		srun --account=cis240473 --partition=shared --time=01:00:00 --cpus-per-task=4 --job-name="testing" --pty bash
		;;
	"ib.bridges2.psc.edu")
		interact -A cis240124p -p RM-shared -t 1:00:00 -n 4
		;;
	"expanse.sdsc.edu")
		srun --partition=shared --pty --account=ucf122 --nodes=1 --ntasks-per-node=4 --time=01:00:00 bash
		;;
	*)
		echo "ALERT: Cluster not found"
		;;
esac
