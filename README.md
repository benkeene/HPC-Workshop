# HPC-Workshop

From within a shell on one of the three following HPC cluster:

- Purdue Anvil
- PSC Bridges-2
- SDSC Expanse

Run the following commands in order:

```shell
source 01_start_session.sh
source 02_python_env.sh
```

After running `02_python_env.sh`, you no longer need to run `02_python_env.sh` again.
Instead, you can run `source activate.sh`.
For example, if tomorrow you wish to log in to the cluster and work on this material again, you can run the following.

```shell
source 01_start_session.sh
source activate.sh
```
