# Codex タスク: llama.cpp に expert router dump を追加 (env-gated, 回帰ゼロ)

## 目的
GLM-5.2 MoE の decode 中、各 token×layer で選ばれた expert id を吐く。
expert 再利用率/working-set を測り、Optane expert paging 戦略の効果を判定するため。

## 実装方針 (最小・非侵襲)
- 既存の `ggml_backend_sched_eval_callback cb_eval` (common.h:451) を使う。新規グラフ改変はしない。
- 対象テンソル: 名前 `ffn_moe_topk` (src/llama-graph.cpp:1460 で cb() 命名済)。
  これは `selected_experts` [n_expert_used, n_tokens] の I32。
- 環境変数 `GLM_DUMP_EXPERTS=<path.jsonl>` が設定されている時だけ有効。未設定なら **完全に no-op (回帰ゼロ)**。
- eval callback 内で、tensor->name が "ffn_moe_topk" に前方一致したら:
  - `ggml_backend_tensor_get` で host にコピー
  - layer 番号は name の suffix か、callback の呼び出し順カウンタで採番
  - 1行 append: `{"layer":L,"experts":[...](全token分flatten or token別)}`
- 参考: `examples/eval-callback/` が cb_eval で tensor 内容を print する既存例。これを流用。

## 配線
- `llama-cli` (tools/main) で、`GLM_DUMP_EXPERTS` がある時 `params.cb_eval` / `cb_eval_user_data` を登録。
- ファイルは追記オープン、プロセス終了で close。

## ビルド (重要・既知の罠)
- conda env: `~/miniforge3/envs/llamacpp-cu131` をアクティベート
- CUDA 13.1 (このenvの nvcc)。host compiler は **gcc-15 必須** (gcc-16 が nvcc 殺す既知問題):
  `cmake ... -DCMAKE_CUDA_HOST_COMPILER=gcc-15`
- SM120: `-DCMAKE_CUDA_ARCHITECTURES=120`
- 既存 build dir `~/src/llama.cpp/build` を使い incremental build (`cmake --build build -j --target llama-cli`)

## 検証
- `GLM_DUMP_EXPERTS=` 未設定で llama-cli が従来通り動く (回帰ゼロ確認)
- 設定時に jsonl が 1 token あたり n_layer 行、各 n_expert_used 個の id で埋まる
- `python ~/projects/glm52-speedup/harness/40_analyze_reuse.py <jsonl>` が走る形式
