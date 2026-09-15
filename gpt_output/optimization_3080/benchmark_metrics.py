"""Decode-only cache hit counts from the unrounded integer log counters."""
import re


def parse_prompt_hits(content):
    token_re = re.compile(r'\[SMoE\] token=(\d+).*gpu_hit_rate=[\d.]+ \((\d+)/(\d+)\)')
    prompt_re = re.compile(r'\[SMoE\] prompt=(-?\d+).*decode_tokens=(\d+)')
    result = {}
    hits = calls = tokens = 0
    for line in content.splitlines():
        match = token_re.search(line)
        if match:
            _, hit, total = map(int, match.groups())
            if not 0 <= hit <= total:
                raise ValueError('invalid GPU hit counters')
            hits += hit
            calls += total
            tokens += 1
        match = prompt_re.search(line)
        if match:
            prompt, expected = map(int, match.groups())
            if tokens != expected:
                raise ValueError(f'prompt {prompt}: {tokens} hit records for {expected} decode tokens')
            if prompt in result:
                raise ValueError(f'duplicate prompt {prompt}')
            result[prompt] = dict(gpu_cache_hits=hits, routed_expert_calls=calls,
                gpu_cache_hit_rate=hits / calls if calls else None)
            hits = calls = tokens = 0
    return result
