# Rules for agents working in this repo

- Do not guess what a script does. Before running any script you are not certain about, read it and work out its effect, including what it writes, overwrites, kills or sends to the cluster. Only then run it.
- Never cancel, kill or resubmit the user's jobs (scancel, kill, pkill, sbatch) unless the user explicitly asks for that exact action in that message. "I need to kick these out" is a plan, not an instruction.
- Never run greatlakes/sync_to_greatlakes.sh while any of the user's jobs are running. It pushes the git-tracked greatlakes/slurm-*.out files over the live job logs and kills the jobs. Check `squeue -u <user>` first; push single files with scp.
- Do not delete or overwrite anything on the cluster or in results directories without the user's approval.
