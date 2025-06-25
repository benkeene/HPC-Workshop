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
