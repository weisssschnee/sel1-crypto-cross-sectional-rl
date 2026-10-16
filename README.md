# 加密货币截面稀疏选股：搜索代码与失败记录

每小时在时点 TOP100 里用公式给币打分，做多前 5、做空后 5，对冲 BTC β。公式由 RL（策略梯度）、GFlowNet 或 GA 在一个小的表达式语法里搜。到 2026-10-16 为止没有候选通过 TRAIN 门；DEV 只在冻结后读过几次，OOS 一次没读。

仓库里只有代码、回执和日志，没有数据面板和模型权重，所以代码能读，不能直接跑。

## 目标函数

定义在 `code/sel1_common.py` 的文件头和 `reward()`，组合扫描在 `code/sel1_sweep.py`。

- 书：信号滞后 1 小时；每边 K=5 等权（both = ±1/10，long = +1/5，short = −1/5）；最短持有 h 小时、带宽 0.5 的状态化持仓；BTC 腿 = −Σ 持仓 × β（720 h 滚动 OLS）；下架币从最后一个真实小时起权重清零。
- 动作：公式 + 组合。组合 = 持有 {4, 24, 72, 168} h × BTC 状态 5 种（全部 / 涨低波 / 涨高波 / 跌低波 / 跌高波，按 BTC 720 h 收益正负和波动分）× 方向 3 种，共 60 个。V2 起策略只出公式，评分时扫全部 60 个组合，取 A 半奖励最高的那个。
- 奖励 t_dd：A 半（奇数月，2021–2025 共 30 个月）月度 cost0 对数收益的 t 值；A 半最大回撤超过回撤线（空世界 q95）时记 0。cost0 = 不扣交易费、扣资金费；验收用 flat6（手续费 5 bp + 滑点 1 bp）。
- 三族：T1 动量 / T2 拥挤 / T3 基差，各有自己的终端子集和盆地 token，每批 128 个公式按 43/43/42 分配；族内 z 分数截断到 ±3 当优势，非法公式 −1。
- 判定（`code/sel1_round.py`）：用 B 半（偶数月）排名，四道门：① t_B > T_null；② t_A > 2 且同号；③ 去掉最有利的 1% 时段后两半 cost0 仍 > 0；④ 至多一年反号。
- 对照世界：NULL = 每个币的信号按固定偏移（≥720 h）循环平移，收益不动；PC = NULL + 植入 P1 = `DELTA(TS_MEAN(log_quote_volume,168),168)`（α 6.908e-5，约 20 bp/天，组合 4/all/both）。ARMS 世界再加 P2（与 P1 相关 −0.0003，α 5.6456e-5，组合 72/all/both）。"找到" = 名人堂或池子里有公式与植入的截面 Spearman 相关 > 0.7。

## 数据

- V4 小时面板：632 币 × 45288 h（2020-11 起），含 TRAIN 期内下架的 94 个币，按时点取 TOP100。
- 11 个终端：log_close, log_high, log_low, log_quote_volume, funding_level, log_oi_value, global_ls_account_ratio, taker_buy_share, premium_close_bps, mark_index_basis_bps, premium_high。
- TRAIN 2021-01..2025-12（A 奇数月 / B 偶数月）；DEV 2026-01..06；OOS 2026-07-01..09-26，不读。
- SEL2（进行中）：约 100 个币 2024–25 的 1 分钟成交分片，聚成小时列（`sel2_minbar.py`，16 列留 10）和小时内统计（`sel2_feat.py`，17 个留 15，覆盖 TOP100 币时的 24.7%）；板块按月对 720 h β 残差相关做层次聚类（`sel2_sector.py`，k=8），相邻月份分组的 ARI 中位数 0.11。

## 目录

- `code/`：77o 上 `runtime\basin3_h1` 的全部 .py / .bat / .ps1（239 个）。SEL1 主干：
  - `sel1_common.py` 世界、书、奖励；`sel1_sweep.py` 60 组合扫描；`sel1_fastts.py` 融合 CUDA 窗口算子（与原算子逐位一致）
  - `sel1_trainer.py` V2 链路用的 PG 训练器（开训后不改）；`sel1_trainer_x.py` / `sel1_trainer_x2.py` 加 `--dedup` / `--pool` / `--gfn` / `--risk-seek` 等开关
  - `sel1_gfn.py`、`sel1_ga.py` / `sel1_ga2.py`、`sel1_pool.py`（组合边际奖励）、`sel1_world2.py`（双植入世界）
  - `sel1_round.py` 判定；`sel1_combo.py` 合成书；`sel1_calib.py` / `sel1_nullplant.py` 植入标定
  - `m2h_*`、`b5_*`、`cr1_*`、`combo_*`、`tv1_*` 是更早的几轮，见 `FAILURE_RECORDS.md`
- `code_m1/`：1 分钟 M1 线。`m1_ruler.py` 是现在仍在用的结算尺子。
- `deps/`：被 import、但不在 `code/` 里的模块。`root_src/okx_alpha_discovery` 有公式 VM（`xs_tscs_vm_v2.py`）、采样器和结算；`root_runtime` 有策略网络构造；`legacy/` 是更早的 alphagpt-rl-dl-base 里的 `model_core`（网络、词表）和 `dso_math.py`。`experiments/` 下的其余模块没带：`rl_v2_autonomous_novelty_search_v4` 只有 `code/sel1_timing_trainer.py`、`code_m1/m2_trainer.py`、`code_m1/s1_vram_stage.py` 和 `deps/root_runtime/train_basin_riskpg_v1.py` 用到；SEL1 训练器只从最后这个文件取网络构造函数，那个 import 在它的 `main()` 里，不会触发。
- `reports/`：`m2h_v1/` 下各轮的 JSON 回执、判定和名人堂（单文件 ≤2 MB），`m1/` 是 M1 线的。
- `logs/`：`SEL1_CHAIN_V2.log`（V2 的 PC / NULL 链路）、`SEL1_ARMS.log`（ARMS v1）。

## 现状（2026-10-16 晚）

- V2：PC 找到 P1（名人堂 500 个里 168 个相关 > 0.7，最好 t_B 5.26），但策略塌缩到少数公式；NULL 停在 checkpoint_0527，等 ARMS 排名。
- ARMS v1（双植入世界，6 种搜索各 3 h）：最终排名 V2 RL > GA > GFN > POOL > DEDUP > FROZEN（随机基线），只有 RL（第 4601 秒）和 GA（第 6474 秒）找回 P1，P2 都没找回。POOL 臂的优势设计有问题（见 `FAILURE_RECORDS.md` 第 12 节），FROZEN 显存溢出、求值次数太少，这两臂不可比。三个搜索器的实现复查都只评"中"，v1 结果只代表这一版实现。修过的 PG / GFN / GA（ARMS v2）10-16 23:05 起跑。

## 实现上踩过的坑

- 族内 z 分数：numpy 对分数全相同的一族算出 std ≈ 1e-16 而不是 0，会给出 ±1 的随机优势；现在 std < 1e-12 时整族记 0（`sel1_dedup.zfam`）。
- 截面 1/99 缩尾会削掉小币暴涨，美化做空；稀疏书一律不缩尾（见 `FAILURE_RECORDS.md` 的 ENR1）。
- 币下架后数据源仍会发成交额为 0 的 K 线；V4 按成交额 > 0 截到最后一个真实小时（`v4_world.py`）。

## 阅读顺序

1. `code/sel1_common.py` 文件头：书和奖励
2. `code/sel1_sweep.py`：组合怎么选
3. `code/sel1_trainer.py`、`code/sel1_trainer_x.py`：采样、优势、更新
4. `code/sel1_round.py`：门
5. `FAILURE_RECORDS.md`
