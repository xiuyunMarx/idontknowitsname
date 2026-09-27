"""Preserve Claude scratchpad evidence and recover logged mixed-run summaries.
No session records or missing common-window completion counts are synthesized.
"""
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = Path('/tmp/claude-1007/-home-xiaoyu-idontknowitsname')
LATEST = '5fd3f06b-1ebf-43ed-9785-032fe725c3a1'
SOURCES = [LATEST, '102d1070-bb96-4068-8499-9e2724387a06']
manifest = []
for session in SOURCES:
    scratch = ROOT / session / 'scratchpad'
    for src in sorted(scratch.glob('mix*')):
        if not src.is_file() or src.suffix not in {'.csv', '.out', '.bash'}:
            continue
        dst = OUT / 'originals' / session / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        assert hashlib.sha256(dst.read_bytes()).hexdigest() == digest
        manifest.append({'source': str(src), 'copy': str(dst.relative_to(OUT)), 'bytes': src.stat().st_size, 'sha256': digest})
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

base = OUT / 'originals' / LATEST
text = (base / 'mix4x4_h10.out').read_text()
programs = {}
windows = {}
for line in text.splitlines():
    m = re.match(r'^\[(mix4x4h10_(.+?))/([^]]+)\] (.*)$', line)
    if m:
        tag, arm, program, fields = m.groups()
        values = dict(x.split('=', 1) for x in fields.split())
        assert (arm, program) not in programs
        programs[arm, program] = dict(arm=arm, tag=tag, program=program, lanes='4', **values)
    m = re.match(r'^\[mix4x4h10_(.+?)\] full-load window ([\d.]+) s, ([\d.]+) sessions/min; (.*)$', line)
    if m:
        arm, window, throughput, counts = m.groups()
        windows[arm] = dict(own_window_s=window, own_throughput=throughput, **dict(x.split('=') for x in counts.split(', ')))
assert 'MIX4X4-DONE' in text
assert len(windows) == 11 and len(programs) == 44
for arm in windows:
    assert {p for a, p in programs if a == arm} == {'BFCL_agent', 'coding_agent', 'doc_analysis', 'fact_check'}

partial_programs = list(csv.DictReader((base / 'mix4x4_partial_programs.csv').open()))
for prior in partial_programs:
    recovered = programs[prior['arm'], prior['program']]
    for key, value in prior.items():
        expected = '' if recovered.get(key) == 'None' else recovered.get(key)
        assert value == expected, (prior['arm'], prior['program'], key, value, expected)

order = ['vanilla', 'kvonly', 'relayout', 'ours', 'continuum', 'cachescout', 'kvflow', 'kvflow_relayout', 'continuum_relayout', 'cachescout_relayout', 'ours_noprefetch']
partial = {r['arm']: r for r in csv.DictReader((base / 'mix4x4_partial.csv').open())}
common_window = min(float(w['own_window_s']) for w in windows.values())
assert all(float(r['window_s']) == common_window for r in partial.values())
rows = []
for arm in order:
    ps = [r for (a, p), r in programs.items() if a == arm]
    calls = sum(int(r['calls']) for r in ps)
    row = dict(arm=arm, tag='mix4x4h10_'+arm, window_s=common_window,
               throughput='', done_window='', **windows[arm],
               sessions=sum(int(r['sessions']) for r in ps), failed=sum(int(r['failed']) for r in ps),
               jct_w_mean_s=round(sum(float(r['jct_mean_s']) for r in ps)/4, 1),
               jct_w_p50_s=round(sum(float(r['jct_p50_s']) for r in ps)/4, 1),
               jct_ref='vanilla', jct_norm=round(sum(float(r['jct_mean_s']) / float(programs['vanilla', r['program']]['jct_mean_s']) for r in ps)/4, 3),
               ttft_w_mean_ms=round(sum(float(r['ttft_mean_ms']) for r in ps)/4),
               ttft_w_p50_ms=round(sum(float(r['ttft_p50_ms']) for r in ps)/4),
               ttft_mean_ms=round(sum(int(r['calls'])*float(r['ttft_mean_ms']) for r in ps)/calls), calls=calls)
    for key in ['device_pct', 'host_pct', 'miss_pct']:
        row[key] = round(sum(int(r['calls'])*float(r[key]) for r in ps)/calls, 1)
    for r in ps:
        name = {'BFCL_agent':'bfcl', 'coding_agent':'coding', 'fact_check':'fact'}.get(r['program'], r['program'])
        for key in ['jct_mean_s', 'jct_p50_s', 'ttft_mean_ms', 'miss_pct', 'host_pct', 'device_pct', 'sessions', 'failed', 'accuracy']:
            row[key+'_'+name] = r[key]
    if arm in partial:
        for key, value in partial[arm].items():
            if key in row and key not in {'throughput', 'done_window'}:
                try:
                    assert float(row[key]) == float(value)
                except ValueError:
                    assert str(row[key]) == value
        row.update(partial[arm])
    row['common_window_counts_recovered'] = arm in partial
    rows.append(row)

def write_csv(name, data):
    fields = list(dict.fromkeys(k for r in data for k in r))
    with (OUT/name).open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)
write_csv('mix4x4_h10_recovered.csv', rows)
write_csv('mix4x4_h10_programs_recovered.csv', [{k: ('' if v == 'None' else v) for k,v in programs[a,p].items()} for a in order for aa,p in programs if aa == a])
checkpoint_rows = []
for row in rows:
    checkpoint = {'arm': row['arm']}
    for seconds in [600, 900, 1200]:
        key = f'done_{seconds}s'
        count = int(row[key])
        assert count == sum(int(p[key]) for (arm, _), p in programs.items() if arm == row['arm'])
        assert seconds <= float(row['own_window_s'])
        checkpoint[key] = count
        checkpoint[f'throughput_{seconds}s_sessions_per_min'] = round(count / (seconds / 60), 4)
    checkpoint_rows.append(checkpoint)
write_csv('mix4x4_throughput_fixed_windows.csv', checkpoint_rows)
interval_rows = []
for row in checkpoint_rows:
    completed = row['done_1200s'] - row['done_600s']
    assert completed >= 0
    interval_rows.append(dict(arm=row['arm'], window_start_s=600,
                              window_end_s=1200, window_duration_s=600,
                              completed_sessions=completed,
                              throughput_sessions_per_min=completed / 10))
write_csv('mix4x4_throughput_600_1200s.csv', interval_rows)

transcript = Path('/home/xiaoyu/.claude/projects/-home-xiaoyu-idontknowitsname') / (LATEST + '.jsonl')
ids = {}
evidence = []
for num, line in enumerate(transcript.open(), 1):
    obj=json.loads(line)
    contents=obj.get('message',{}).get('content',[])
    if not isinstance(contents,list):
        continue
    for c in contents:
        if c.get('type')=='tool_use' and 'mix4x4' in json.dumps(c.get('input',{})):
            ids[c['id']]={'line':num, 'input':c.get('input',{})}
        if c.get('type')=='tool_result' and c.get('tool_use_id') in ids:
            evidence.append({'source':str(transcript), 'result_line':num, 'call':ids[c['tool_use_id']], 'result':c.get('content')})
(OUT/'claude_mix4x4_tool_evidence.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
print('Copied original files:',len(manifest))
print('Recovered arms:',len(rows),'program rows:',len(programs))
print('Validated all',len(partial_programs),'program rows and',len(partial),'arm rows against original partial CSVs.')
print('Common-window completion counts available for',len(partial),'of 11 arms; missing values remain blank.')
