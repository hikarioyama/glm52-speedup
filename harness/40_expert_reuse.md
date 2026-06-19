# 40 — expert 再利用率 / working-set 測定 (勝ち筋の生命線)

戦略2(Optane expert paging)が効くか = **生成中に実際に触る expert が 254GB のうちどれだけ偏るか**で決まる。
偏り大 → hot expert を VRAM 常駐、cold 裾だけ paging → overlap で完全隠蔽 = 勝ち。

## 測定にはllama.cpp の小パッチが要る (Codex 委譲)
router の top-k 選択結果(layer, token, expert_ids[])を吐く。場所:
- MoE 経路の `ggml_mul_mat_id` に渡る `selected_experts` / `ids` テンソル
- 環境変数 `GLM_DUMP_EXPERTS=path.jsonl` ガードで、off 時は完全に回帰ゼロ
- 各 decode step で `{"tok":i,"layer":l,"experts":[...]}` を1行 append

## 解析 (パッチ後)
`python 40_analyze_reuse.py out/expert_ids.jsonl`
出力:
- per-layer ユニーク expert 数 / 総 expert 数 (= working-set 比)
- 累積カバレッジ曲線: top-N hot expert で活性化の何%をカバーできるか
- → 「VRAM に乗る expert 数(192GB-非expert分)で活性の何%カバー = paging率」を直接算出
- これが paging hit 率の理論値。低 paging 率なら戦略2で天井 ~130 tok/s に肉薄可能と判断
