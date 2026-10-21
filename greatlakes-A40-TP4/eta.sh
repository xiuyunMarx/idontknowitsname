#!/bin/bash
# One-shot queue estimate for a TP=4 job (run locally, talks to the cluster once; never loop it).
# SLURM only computes StartTime for the top bf_job_part_count_reserve=10 pending jobs per partition, so for
# ours it stays Unknown; this script estimates it from running jobs' end times instead: for every spgpu node,
# when do CPU/memory/GPU free up enough for the request. It is a lower bound (higher-priority jobs may take
# the slot first) and ignores reservations/planned backfills, so read it with the rank line.
#   ./greatlakes-A40-TP4/eta.sh                 the configured request (16 cpu, 160G, 4 gpu)
#   ./greatlakes-A40-TP4/eta.sh 8 96 4          a hypothetical smaller request
set -euo pipefail
cd "$(dirname "$0")/.."
source greatlakes/config.sh; source greatlakes-A40-TP4/config.sh
CPU=${1:-$GL_CPUS}; MEM=${2:-${GL_MEM%G}}; GPU=${3:-${GL_GRES##*:}}
ssh gl "
echo '== our pending jobs (rank among eligible pending in $GL_PARTITION, by priority)'
squeue -p $GL_PARTITION -t PD -h -o '%i %Q %r %u' | awk '\$3==\"Priority\"||\$3==\"Resources\"' | sort -k2 -nr \
  | awk '{print NR, \$0}' | awk -v me=\$USER '\$5==me || NR<=3 {print \"  rank \" \$1 \": job \" \$2 \" prio \" \$3 \" \" \$4 \" \" \$5}'
squeue -u \$USER -h -o '  %i %T start=%S reason=%r' --start
echo '== earliest slot per node for cpu=$CPU mem=${MEM}G gpu=$GPU (from running jobs end times)'
scontrol show job -o | grep JobState=RUNNING | grep Partition=$GL_PARTITION > /tmp/eta_j.\$\$
scontrol show node -o \$(sinfo -p $GL_PARTITION -N -h -o %N | paste -sd,) > /tmp/eta_n.\$\$
# nodes inside an active reservation that does not admit our account or user are not ours to backfill into
scontrol show res -o | grep State=ACTIVE | grep -vE 'Accounts=\S*$GL_ACCOUNT|Users=\S*'\$USER \
  | grep -oE 'Nodes=\S+' | cut -d= -f2 | xargs -r -n1 scontrol show hostnames > /tmp/eta_r.\$\$
python3 - /tmp/eta_j.\$\$ /tmp/eta_n.\$\$ $CPU $MEM $GPU /tmp/eta_r.\$\$ <<'PY'
import sys, re, datetime as dt
def kv(line, k, d=None):
    m = re.search(r'(?:^|\s)' + k + r'=(\S+)', line); return m.group(1) if m else d
def tres(s):
    return dict(p.split('=', 1) for p in (s or '').split(',') if '=' in p)
def mem(s):
    m = re.match(r'([\d.]+)([KMGT])?', s or '0'); return float(m.group(1)) * {'K': 1/1024, 'M': 1, 'G': 1024, 'T': 1024**2}[m.group(2) or 'M']
nc, nm, ng = int(sys.argv[3]), int(sys.argv[4]) * 1024, int(sys.argv[5]); now = dt.datetime.now()
reserved = set(open(sys.argv[6]).read().split())
nodes = {}
for l in open(sys.argv[2]):
    if kv(l, 'NodeName') in reserved: continue
    a = tres(kv(l, 'AllocTRES', ''))
    nodes[kv(l, 'NodeName')] = [int(kv(l, 'CPUTot')) - int(a.get('cpu', 0)), float(kv(l, 'RealMemory')) - mem(a.get('mem')), 8 - int(a.get('gres/gpu', 0)), kv(l, 'State'), []]
for l in open(sys.argv[1]):
    n = kv(l, 'NodeList')
    if n in nodes:
        t = tres(kv(l, 'AllocTRES', '')); e = dt.datetime.strptime(kv(l, 'EndTime')[:19], '%Y-%m-%dT%H:%M:%S')
        nodes[n][4].append((e, int(t.get('cpu', 0)), mem(t.get('mem')), int(t.get('gres/gpu', 0))))
out = []
for n, (fc, fm, fg, st, ends) in nodes.items():
    t = now
    for e, c, m, g in sorted(ends):
        if fc >= nc and fm >= nm and fg >= ng: break
        t = e; fc += c; fm += m; fg += g
    if fc >= nc and fm >= nm and fg >= ng: out.append((t, n, st))
print('  (%d nodes usable, %d in reservations closed to us)' % (len(nodes), len(reserved)))
for t, n, st in sorted(out)[:6]:
    print('  %s %-16s %s (in %.1f h)' % (n, st, t.strftime('%m-%d %H:%M'), (t - now).total_seconds() / 3600))
PY
rm -f /tmp/eta_j.\$\$ /tmp/eta_n.\$\$ /tmp/eta_r.\$\$
" < /dev/null
