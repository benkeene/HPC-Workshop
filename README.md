# HPC-Workshop

## ACCESS

[ACCESS Website](https://access-ci.org/)

[ACCESS Allocations Page](https://allocations.access-ci.org/)

Our project: CIS240473

View your resource username(s) on the rightmost column of our project.

## Purdue

[Anvil user guide](https://www.rcac.purdue.edu/knowledge/anvil)

[Open OnDemand](https://ondemand.anvil.rcac.purdue.edu/)

## PSC

[Bridges-2 user guide](https://www.psc.edu/resources/bridges-2/user-guide/)

[PSC password change utility](https://apr.psc.edu/)

[Open OnDemand](https://ondemand.bridges2.psc.edu/)

## SDSC

[Expanse user guide](https://www.sdsc.edu/systems/expanse/user_guide.html)

[Open OnDemand](https://portal.expanse.sdsc.edu/)

# Usage

From within a shell on one of the three following HPC clusters:

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

# Github SSH

https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

Add to your `.ssh/config`

```shell
host github.com
        User git
        Hostname github.com
        PreferredAuthentications publickey
        IdentityFile ~/.ssh/id_github
```
