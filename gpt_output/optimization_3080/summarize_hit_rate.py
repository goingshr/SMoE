"""Associate raw decode hit counts with completed prompts and policies."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics

p = argparse.ArgumentParser()
p.add_argument('log', type=Path)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
token_re = re.compile(r'\[SMoE\] token=(\d+).*gpu_hit_rate=[\d.]+ \((\d+)/(\d+)\)')
prompt_re = re.compile(r'\[SMoE\] prompt=(-?\d+).*avg_decode=([\d.]+) s.*decode_tokens=(\d+)')
records, tokens = [], []
for line in a.log.read_text().splitlines():
    token = token_re.search(line)
    if token:
        tokens.append(tuple(map(int, token.groups())))
        continue
    prompt = prompt_re.search(line)
    if prompt:
        i, seconds, count = prompt.groups()
        row = dict(policy='legacy', prompt=int(i), decode_s=float(seconds), decode_tokens=int(count))
    elif line.startswith('[SWEEP] ') and '{' in line:
        _, policy, payload = line.split(' ', 2)
        data = json.loads(payload)
        row = {k: data[k] for k in ('prompt', 'decode_s', 'decode_tokens')}
        row['policy'] = policy
    else:
        continue
    assert len(tokens) == row['decode_tokens'], (row, len(tokens))
    hits, total = sum(t[1] for t in tokens), sum(t[2] for t in tokens)
    row.update(hits=hits, expert_calls=total, mean_hit_rate=hits/total if total else None,
               max_token_hit_rate=max((t[1]/t[2] for t in tokens), default=None))
    records.append(row)
    tokens = []

grouped = defaultdict(list)
for row in records:
    if row['prompt'] >= 0:
        grouped[row['policy']].append(row)
summaries = []
for policy, rows in grouped.items():
    summaries.append(dict(policy=policy, measured_prompts=len(rows),
        mean_decode_s=statistics.fmean(r['decode_s'] for r in rows),
        mean_prompt_hit_rate=statistics.fmean(r['mean_hit_rate'] for r in rows),
        weighted_hit_rate=sum(r['hits'] for r in rows)/sum(r['expert_calls'] for r in rows),
        max_token_hit_rate=max(r['max_token_hit_rate'] for r in rows)))
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(dict(prompts=records, summary=summaries), indent=2))
print(json.dumps(summaries, indent=2))
