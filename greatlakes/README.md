# Running the plan on Great Lakes

The experiment is `plan_on_greatlake.md`: 12 arms, each with four per-agent concurrency sweeps
(18 levels) and five mixed-workload lane levels, Qwen3-32B-AWQ on one A40 48 GB (partition spgpu), device pool
31830 tokens, host pool 20 GB. One SLURM job per arm starts the arm's server once and runs every
level on it (`benchmark/run_arm.py`); the only cache resets inside a job are one KV flush before the
mixed block and one between mixed levels (`POST /v1/cache/flush`, all arms). The jobs are
independent and run in parallel wherever SLURM finds a free A40.

## 1. Local machine: sync

```
./greatlakes/sync_to_greatlakes.sh <uniqname> --dry-run    # see what goes
./greatlakes/sync_to_greatlakes.sh <uniqname>
```

Sends the repo including the two git-ignored search caches (HoVer `wiki_cache`, BFCL `web_cache`),
so the agents never query Wikipedia from a compute node. Build caches, bytecode, the paper and
`benchmark/results/` are not sent.

## 2. Login node: build the environment

`greatlakes/config.sh` holds the account, partition, gres, modules and scratch path. To check them
against the cluster:

```
sinfo -o "%P %G %l %D %m %c" | grep -i gpu          # partitions, gres, max wall time, cores
sacctmgr show assoc user=$USER format=account
module avail cuda gcc
```

Then, once (idempotent, needs internet, no GPU):

```
bash greatlakes/setup_env.sh
```

It installs miniforge if needed, creates the `mortis` env (python 3.12, conda-forge gcc 12, torch 2.9.1
cu128, `pip install -e sglang/python`), downloads Zig 0.16.0, builds `jac` from `jaseci/jac`
(`zig build -Dno-ninja -Ddev`), installs byllm's deps into jac's runtime, downloads the Qwen3-32B-AWQ weights
into `HF_HOME`, warms the static-analysis cache, and finally profiles the model on one A40 through a
blocking `srun` (step 6: `./start_server.bash profile`, the same engine configuration as the arms).
The profile, `model/Qwen--Qwen3-32B-AWQ.profile.json`, holds the device rates the planner
schedules with: prefill tokens/s idle and under decode load, and the host-to-device promote rate
(the H2D bandwidth in tokens/s). That first GPU start also JIT-compiles sglang's kernels. Re-measure
with `REPROFILE=1 bash greatlakes/setup_env.sh`.

## 3. Submit

```
./greatlakes/submit_all.sh                 # prepare job, then 12 arm jobs with afterok on it
./greatlakes/submit_all.sh ours vanilla    # a subset
squeue -u $USER
```

`prepare.sbatch` is the end-to-end check: a short fact_check level plus a flush on vanilla and on
cachescout, through the real jac -> server path. It refuses to run without the profile from setup
step 6. Check `greatlakes/slurm-mortis-prepare-<id>.out` and
`benchmark/results/vanilla/smoke_fact_check/summary.json`; the arm jobs wait for it (afterok).

Each arm job asks for 1 A40, 12 cores, 64 GB, 72 h (config.sh). A job first copies the model to the
node's local disk and points `JAC_DATA_PATH` (jac's per-program sqlite memory) there too: on the NFS
home, concurrent sessions from several nodes corrupt that database (`database disk image is malformed`). Estimated 40 to 60 h per arm (about
3x the local 14B/3090 runs: 2.3x the model, 0.75x the memory bandwidth); resume if it runs out.

## 4. Watch, resume

```
tail -f benchmark/results/<arm>/progress.log         # one line per level: throughput, cache split
tail -f greatlakes/slurm-mortis-<arm>-<id>.out       # the driver's full output
tail -f benchmark/results/<arm>/server.0.log         # the server
```

A job that died or hit its wall time is resumed by submitting the same arm again; finished levels
(those with `summary.json`) are skipped, the unfinished one is redone:

```
PREP=none ./greatlakes/submit_all.sh <arm>
```

To rerun specific levels, delete their directories first, or pass `--only`:

```
sbatch --account=... --partition=... --gres=... --cpus-per-task=24 --mem=128G --time=12:00:00 \
    --job-name=mortis-ours greatlakes/arm.sbatch ours --only 'mix_L*'
```

## 5. Results

Everything stays in the repo under `benchmark/results/<arm>/` (git-ignored by the `results/` rule):

| path | content |
|---|---|
| `<level>/summary.json` | the level's metrics: JCT, TTFT, device/host/miss %, sessions/min, prompt/recompute/output ktokens/min |
| `<level>/sessions.jsonl`, `session_events.jsonl` | one record per session: program, input, pid, t0, t1, verdict |
| `<level>/logs/*.log` | every session's jac output |
| `<level>/server.log` | the server's log slice for this level (`[serve]`, `[decode]`, `[call]`, `[promote]`, sglang batch stats) |
| `<level>/requests.jsonl` | the request trace slice: request_admitted, request_start, first_token (cache split, ttft), request_end (output tokens) |
| `server.<n>.log`, `requests.jsonl` | the whole server, n = start count (SERVER-RESTART in progress.log) |
| `progress.log`, `plan.json`, `git_head.txt`, `working_tree.patch` | progress and provenance |

Levels are `sweep_<program>_c<c>` and `mix_L<lanes>`. Pull them back and build the tables:

```
./greatlakes/pull_results.sh            # uses the `gl` ssh alias; one Duo prompt per 72h
python -m benchmark.summarize          # benchmark/results/sweep_<program>.csv, mixed*.csv
```

Throughput in those tables: `prompt_ktokens_per_min` counts every request's full prompt,
`recompute_ktokens_per_min` only the uncached part (the prefill the GPU did), per minute of the
level (sweeps) or of the progress window (mixed_steady.csv).
