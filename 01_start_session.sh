#!/bin/bash

CLUSTER=$(hostname -d)
echo $CLUSTER

case $CLUSTER in
"anvil.rcac.purdue.edu")
	srun --account=cis240473 --partition=shared --time=01:00:00 --ntasks-per-node=4 --job-name="testing" --pty bash
	;;
"ib.bridges2.psc.edu")
	interact -A cis240124p -p RM-shared -t 1:00:00 -n 16
	;;
"expanse.sdsc.edu")
	srun --partition=shared --pty --account=ucf122 --nodes=1 --ntasks-per-node=4 --time=01:00:00 bash
	;;
*)
	echo "ALERT: Cluster not found"
	;;
esac
