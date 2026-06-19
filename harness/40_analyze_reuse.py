#!/usr/bin/env python3
"""expert_ids.jsonl を読み、working-set / 累積カバレッジを算出。
各行: {"tok":int,"layer":int,"experts":[int,...]}
"""
import sys, json
from collections import Counter, defaultdict

def main(path):
    layer_counts = defaultdict(Counter)   # layer -> expert_id -> 活性回数
    layer_total = defaultdict(int)
    n_tok = 0
    toks = set()
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        toks.add(r["tok"])
        for e in r["experts"]:
            layer_counts[r["layer"]][e] += 1
            layer_total[r["layer"]] += 1
    n_tok = len(toks)
    print(f"tokens={n_tok} layers={len(layer_counts)}")
    # 全層集計
    global_unique, global_total = 0, 0
    for l in sorted(layer_counts):
        c = layer_counts[l]
        uniq = len(c)
        tot = layer_total[l]
        global_unique += uniq
        global_total += tot
        # この層で活性の80/90/95%をカバーするのに要る上位 expert 数
        acts = sorted(c.values(), reverse=True)
        csum, need = 0, {80: None, 90: None, 95: None}
        for i, v in enumerate(acts, 1):
            csum += v
            for thr in need:
                if need[thr] is None and csum >= tot * thr / 100:
                    need[thr] = i
        print(f"L{l:>3}: unique={uniq:>4}  cover80={need[80]} cover90={need[90]} cover95={need[95]} (of activations {tot})")
    print(f"\n=== 全層: ユニーク活性 expert(層×expert) = {global_unique} ===")
    print("→ working-set 比 = unique / (n_layer * n_expert_total)。")
    print("→ cover90 の合計 expert を VRAM 常駐できれば paging率 ~10%。これと VRAM 予算を突き合わせる。")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/expert_ids.jsonl")
