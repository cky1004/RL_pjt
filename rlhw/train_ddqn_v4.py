#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_ddqn_v4.py (FINAL)

- 데이터: train.csv / test.csv (+ split_meta.json 권장)
- baseline: fixed_k(운영) + best_fixed(utility 기준)
- RL: DDQN으로 그룹별 k를 동적으로 선택
- 목표: posF1(불량 검출) ↑, alarm_rate ↓ (trade-off 개선)
- 안정성: spike penalty(알람 급증 억제) + best-utility checkpoint 저장

출력 파일
- ddqn_model_last.pt   : 마지막 모델
- ddqn_model_best.pt   : Utility 최고(=posF1 - lam*alarm) 모델
- eval_log.csv         : 평가 로그
- learning_curve.png   : 학습 곡선(dual axis)
- results_summary.json : 제출용 요약

실행 예시 (Windows PowerShell)
python train_ddqn_v4.py --train_csv train.csv --test_csv test.csv --meta_json split_meta.json --episodes 600 --eval_every 20 --device cpu ^
  --sigma_mode value --k_min 2 --k_max 7 --k_step 0.5 ^
  --fixed_k 6 --lam_alarm 0.8 --baseline_metric utility ^
  --fp_pen 0.30 --fn_pen 1.00 --alarm_pen 0.03 ^
  --spike_coef 200 --cap_source best_fixed
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Dict, Any, Optional, Callable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from env_dynamic_band import DynamicBandEnv, ActionSpec, make_action_table, pos_f1_from_counts


# ----------------------
# Replay Buffer
# ----------------------
@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.buf: Deque[Transition] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def push(self, tr: Transition):
        self.buf.append(tr)

    def sample(self, batch: int) -> List[Transition]:
        idx = self.rng.choice(len(self.buf), size=batch, replace=False)
        arr = list(self.buf)
        return [arr[i] for i in idx]

    def __len__(self) -> int:
        return len(self.buf)


# ----------------------
# Q Network
# ----------------------
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def greedy(q: QNet, obs: np.ndarray, device: str) -> int:
    x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    return int(torch.argmax(q(x), dim=1).item())


def set_all_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic(가능한 범위)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eval_over_all_groups(
    env: DynamicBandEnv,
    policy_fn: Callable[[np.ndarray], int],
) -> Dict[str, float]:
    """
    테스트셋 전체 그룹을 한 에피소드로 평가 (deterministic).
    """
    # 전체 그룹을 한 번에
    env.episode_len = len(env.group_keys)
    env.random_start = False

    obs, _ = env.reset(options={"start_index": 0})
    done = False

    tp=fp=fn=tn=0
    alarm_points=0
    total=0
    tot_return=0.0

    while not done:
        a = int(policy_fn(obs))
        obs, r, done, _, info = env.step(a)
        tot_return += float(r)

        tp += int(info["tp"]); fp += int(info["fp"]); fn += int(info["fn"]); tn += int(info["tn"])
        alarm_points += int(info["alarm_points"])
        total += int(info["n"])

    posf1 = pos_f1_from_counts(tp, fp, fn)
    micro_acc = float((tp + tn) / max(1, total))
    alarm_rate = float(alarm_points / max(1, total))
    return dict(
        return_total=float(tot_return),
        posF1_global=float(posf1),
        micro_acc=float(micro_acc),
        alarm_rate=float(alarm_rate),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), total=int(total),
    )


def compute_utility(posF1: float, alarm_rate: float, lam_alarm: float) -> float:
    return float(posF1 - lam_alarm * alarm_rate)


def sweep_fixed_k(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cols_map: Dict[str, str],
    args: argparse.Namespace,
    alarm_cap_for_env: Optional[float] = None,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    action grid 전체를 훑어서 best-fixed k를 선택.
    baseline_metric:
      - "posF1": posF1_global 최대
      - "utility": posF1_global - lam_alarm*alarm_rate 최대
    """
    action_table = make_action_table(args.k_min, args.k_max, args.k_step)

    best_k = action_table[0].k
    best_score = -1e18
    best_train = {}
    best_test = {}

    for spec in action_table:
        env_tr = DynamicBandEnv(
            df_train, cols_map=cols_map,
            episode_len=args.episode_len, random_start=False, seed=args.seed,
            poly_deg=args.poly_deg, sigma_mode=args.sigma_mode, sigma_floor=args.sigma_floor,
            fp_pen=args.fp_pen, fn_pen=args.fn_pen, alarm_pen=args.alarm_pen, switch_cost=0.0,
            alarm_cap=alarm_cap_for_env, spike_coef=0.0,
            k_min=args.k_min, k_max=args.k_max, k_step=args.k_step,
        )
        env_te = DynamicBandEnv(
            df_test, cols_map=cols_map,
            episode_len=args.episode_len, random_start=False, seed=args.seed,
            poly_deg=args.poly_deg, sigma_mode=args.sigma_mode, sigma_floor=args.sigma_floor,
            fp_pen=args.fp_pen, fn_pen=args.fn_pen, alarm_pen=args.alarm_pen, switch_cost=0.0,
            alarm_cap=alarm_cap_for_env, spike_coef=0.0,
            k_min=args.k_min, k_max=args.k_max, k_step=args.k_step,
        )
        # fixed policy index
        a = env_tr.action_table.index(spec)

        tr = eval_over_all_groups(env_tr, lambda ob, aa=a: aa)
        te = eval_over_all_groups(env_te, lambda ob, aa=a: aa)

        if args.baseline_metric == "posF1":
            score = te["posF1_global"]
        else:
            score = compute_utility(te["posF1_global"], te["alarm_rate"], args.lam_alarm)

        if score > best_score:
            best_score = score
            best_k = spec.k
            best_train = tr
            best_test = te

    return float(best_k), best_train, best_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--meta_json", default="split_meta.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)

    # environment/model options
    ap.add_argument("--poly_deg", type=int, default=3)
    ap.add_argument("--sigma_mode", type=str, default="value", choices=["value", "resid"])
    ap.add_argument("--sigma_floor", type=float, default=0.01)

    # action grid
    ap.add_argument("--k_min", type=float, default=2.0)
    ap.add_argument("--k_max", type=float, default=7.0)
    ap.add_argument("--k_step", type=float, default=0.5)

    # episode
    ap.add_argument("--episode_len", type=int, default=60)

    # reward weights
    ap.add_argument("--fp_pen", type=float, default=0.10)
    ap.add_argument("--fn_pen", type=float, default=5.00)
    ap.add_argument("--alarm_pen", type=float, default=0.00)
    ap.add_argument("--switch_cost", type=float, default=0.02)

    # training
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--buffer", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--warmup_steps", type=int, default=3000)
    ap.add_argument("--target_update", type=int, default=200)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.01)
    ap.add_argument("--eps_decay_steps", type=int, default=20000)
    ap.add_argument("--eval_every", type=int, default=20)

    # baseline/selection
    ap.add_argument("--fixed_k", type=float, default=6.0)
    ap.add_argument("--baseline_metric", type=str, default="utility", choices=["utility", "posF1"])
    ap.add_argument("--lam_alarm", type=float, default=0.8)

    # spike suppression
    ap.add_argument("--cap_source", type=str, default="best_fixed", choices=["best_fixed", "fixed", "none"])
    ap.add_argument("--cap_value", type=float, default=0.005)  # 0.5%
    ap.add_argument("--spike_coef", type=float, default=200.0)

    args = ap.parse_args()

    set_all_seeds(args.seed)

    df_train = pd.read_csv(args.train_csv)
    df_test = pd.read_csv(args.test_csv)

    # meta cols mapping (if exists)
    cols_map = None
    try:
        with open(args.meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cols_map = meta.get("cols", None)
    except Exception:
        meta = None
        cols_map = None

    # normalize cols_map keys if present
    if cols_map is not None:
        # expected keys: EQUIPMENTID, GROUPNUM, VALUE, PRED
        cols_map = {
            "EQUIPMENTID": cols_map.get("EQUIPMENTID", "EQUIPMENTID"),
            "GROUPNUM": cols_map.get("GROUPNUM", "GROUPNUM"),
            "VALUE": cols_map.get("VALUE", "VALUE"),
            "PRED": cols_map.get("PRED", "PRED"),
        }

    # fixed_k baseline
    fixed_spec = ActionSpec(k=float(args.fixed_k))
    action_table = make_action_table(args.k_min, args.k_max, args.k_step)
    if fixed_spec not in action_table:
        raise ValueError(f"fixed_k={args.fixed_k} not in [{args.k_min}..{args.k_max}] step {args.k_step}")

    # best-fixed (for comparison and cap source)
    best_k, best_fixed_train, best_fixed_test = sweep_fixed_k(df_train, df_test, cols_map, args, alarm_cap_for_env=None)

    # alarm cap for spike suppression
    if args.cap_source == "none":
        alarm_cap = None
    elif args.cap_source == "fixed":
        alarm_cap = float(args.cap_value)
    else:
        alarm_cap = float(best_fixed_test["alarm_rate"])

    # envs for training/eval
    env_train = DynamicBandEnv(
        df_train, cols_map=cols_map,
        episode_len=args.episode_len, random_start=True, seed=args.seed,
        poly_deg=args.poly_deg, sigma_mode=args.sigma_mode, sigma_floor=args.sigma_floor,
        fp_pen=args.fp_pen, fn_pen=args.fn_pen, alarm_pen=args.alarm_pen, switch_cost=args.switch_cost,
        alarm_cap=alarm_cap, spike_coef=args.spike_coef,
        k_min=args.k_min, k_max=args.k_max, k_step=args.k_step,
    )
    env_test = DynamicBandEnv(
        df_test, cols_map=cols_map,
        episode_len=len(df_test.groupby([cols_map["EQUIPMENTID"], cols_map["GROUPNUM"]]).ngroups) if cols_map else args.episode_len,
        random_start=False, seed=args.seed,
        poly_deg=args.poly_deg, sigma_mode=args.sigma_mode, sigma_floor=args.sigma_floor,
        fp_pen=args.fp_pen, fn_pen=args.fn_pen, alarm_pen=args.alarm_pen, switch_cost=0.0,
        alarm_cap=alarm_cap, spike_coef=0.0,  # 평가에서 spike penalty는 OFF
        k_min=args.k_min, k_max=args.k_max, k_step=args.k_step,
    )

    obs_dim = int(env_train.observation_space.shape[0])
    n_actions = int(env_train.action_space.n)
    device = args.device

    q = QNet(obs_dim, n_actions).to(device)
    q_tgt = QNet(obs_dim, n_actions).to(device)
    q_tgt.load_state_dict(q.state_dict())
    q_tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    buf = ReplayBuffer(args.buffer, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    # fixed/best-fixed eval (deterministic, full test)
    env_fixed = DynamicBandEnv(
        df_test, cols_map=cols_map,
        episode_len=len(env_test.group_keys), random_start=False, seed=args.seed,
        poly_deg=args.poly_deg, sigma_mode=args.sigma_mode, sigma_floor=args.sigma_floor,
        fp_pen=args.fp_pen, fn_pen=args.fn_pen, alarm_pen=args.alarm_pen, switch_cost=0.0,
        alarm_cap=None, spike_coef=0.0,
        k_min=args.k_min, k_max=args.k_max, k_step=args.k_step,
    )
    fixed_a = env_fixed.action_table.index(fixed_spec)
    fixed_test = eval_over_all_groups(env_fixed, lambda ob, aa=fixed_a: aa)

    best_spec = ActionSpec(k=float(best_k))
    best_a = env_fixed.action_table.index(best_spec)
    best_test = eval_over_all_groups(env_fixed, lambda ob, aa=best_a: aa)

    # print baselines
    print(f"[BASE fixed k={args.fixed_k}]  posF1={fixed_test['posF1_global']:.5f}  alarm={fixed_test['alarm_rate']:.5f}  util={compute_utility(fixed_test['posF1_global'], fixed_test['alarm_rate'], args.lam_alarm):.5f}")
    print(f"[BASE best-fixed k={best_k}] posF1={best_test['posF1_global']:.5f}  alarm={best_test['alarm_rate']:.5f}  util={compute_utility(best_test['posF1_global'], best_test['alarm_rate'], args.lam_alarm):.5f}")
    print(f"[CAP] source={args.cap_source}  alarm_cap={None if alarm_cap is None else round(alarm_cap,6)}  spike_coef={args.spike_coef}")

    eval_rows: List[Dict[str, Any]] = []
    train_returns: List[float] = []
    global_step = 0

    best_utility = -1e18
    best_ckpt = None

    for ep in range(1, args.episodes + 1):
        obs, _ = env_train.reset()
        done = False
        ep_ret = 0.0

        while not done:
            # epsilon schedule
            eps = args.eps_end + (args.eps_start - args.eps_end) * np.exp(-global_step / max(1, args.eps_decay_steps))
            if rng.random() < eps:
                a = int(env_train.action_space.sample())
            else:
                a = greedy(q, obs, device)

            obs2, r, done, _, _info = env_train.step(a)
            buf.push(Transition(obs, a, float(r), obs2, bool(done)))
            obs = obs2
            ep_ret += float(r)
            global_step += 1

            if len(buf) >= max(args.batch, args.warmup_steps):
                batch = buf.sample(args.batch)
                s = torch.tensor(np.stack([t.s for t in batch]), dtype=torch.float32, device=device)
                a_t = torch.tensor([t.a for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
                r_t = torch.tensor([t.r for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
                s2 = torch.tensor(np.stack([t.s2 for t in batch]), dtype=torch.float32, device=device)
                d = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device).unsqueeze(1)

                q_sa = q(s).gather(1, a_t)
                with torch.no_grad():
                    a2 = torch.argmax(q(s2), dim=1, keepdim=True)
                    q_next = q_tgt(s2).gather(1, a2)
                    target = r_t + args.gamma * (1.0 - d) * q_next

                loss = nn.SmoothL1Loss()(q_sa, target)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                if global_step % args.target_update == 0:
                    q_tgt.load_state_dict(q.state_dict())

        train_returns.append(ep_ret)

        if ep % args.eval_every == 0:
            stats = eval_over_all_groups(env_test, lambda ob: greedy(q, ob, device))
            util = compute_utility(stats["posF1_global"], stats["alarm_rate"], args.lam_alarm)

            row = {"episode": ep, "train_return": ep_ret, "utility": util, **stats}
            eval_rows.append(row)

            print(
                f"[EP {ep:04d}] ret={ep_ret:.3f} | posF1={stats['posF1_global']:.5f} | "
                f"alarm={stats['alarm_rate']:.5f} | util={util:.5f}"
            )

            # best checkpoint by utility
            if util > best_utility:
                best_utility = util
                best_ckpt = {
                    "state_dict": q.state_dict(),
                    "obs_dim": obs_dim,
                    "n_actions": n_actions,
                    "action_table": [a.k for a in env_train.action_table],
                    "args": vars(args),
                    "best_utility": float(best_utility),
                    "best_episode": int(ep),
                }
                torch.save(best_ckpt, "ddqn_model_best.pt")

    # save last
    torch.save(
        {
            "state_dict": q.state_dict(),
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "action_table": [a.k for a in env_train.action_table],
            "args": vars(args),
        },
        "ddqn_model_last.pt",
    )

    # save logs/plots/summary
    if eval_rows:
        df_log = pd.DataFrame(eval_rows)
        df_log.to_csv("eval_log.csv", index=False, encoding="utf-8")
        print("Saved: eval_log.csv")

    # plot (dual axis)
    if eval_rows:
        xs = [r["episode"] for r in eval_rows]
        pos = [r["posF1_global"] for r in eval_rows]
        alarm = [r["alarm_rate"] for r in eval_rows]
        util = [r["utility"] for r in eval_rows]

        fig, ax1 = plt.subplots()
        ax1.plot(range(1, len(train_returns)+1), train_returns, label="train return")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Return")

        ax2 = ax1.twinx()
        ax2.plot(xs, pos, marker="o", label="test posF1")
        ax2.plot(xs, alarm, marker="s", label="test alarm_rate")
        ax2.set_ylabel("posF1 / alarm_rate")

        # baselines
        ax2.axhline(fixed_test["posF1_global"], linestyle="--", label=f"fixed k={args.fixed_k} posF1")
        ax2.axhline(fixed_test["alarm_rate"], linestyle=":", label=f"fixed k={args.fixed_k} alarm")
        ax2.axhline(best_test["posF1_global"], linestyle="--", label=f"best-fixed k={best_k} posF1")
        ax2.axhline(best_test["alarm_rate"], linestyle=":", label=f"best-fixed k={best_k} alarm")

        # legend merge
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc="center right")

        fig.tight_layout()
        fig.savefig("learning_curve.png", dpi=150)
        plt.close(fig)
        print("Saved: learning_curve.png")

        # utility curve alone (PPT용)
        plt.figure()
        plt.plot(xs, util, marker="o", label="test utility = posF1 - lam*alarm")
        plt.axhline(compute_utility(fixed_test["posF1_global"], fixed_test["alarm_rate"], args.lam_alarm), linestyle="--", label=f"fixed k={args.fixed_k} utility")
        plt.axhline(compute_utility(best_test["posF1_global"], best_test["alarm_rate"], args.lam_alarm), linestyle="--", label=f"best-fixed k={best_k} utility")
        plt.xlabel("Episode")
        plt.ylabel("Utility")
        plt.legend()
        plt.tight_layout()
        plt.savefig("utility_curve.png", dpi=150)
        plt.close()
        print("Saved: utility_curve.png")

    # summary JSON
    final_stats = eval_over_all_groups(env_test, lambda ob: greedy(q, ob, device))
    final_util = compute_utility(final_stats["posF1_global"], final_stats["alarm_rate"], args.lam_alarm)

    summary = {
        "meta": meta,
        "config": vars(args),
        "baselines": {
            f"fixed_k_{args.fixed_k}": {**fixed_test, "utility": compute_utility(fixed_test["posF1_global"], fixed_test["alarm_rate"], args.lam_alarm)},
            f"best_fixed_k_{best_k}": {**best_test, "utility": compute_utility(best_test["posF1_global"], best_test["alarm_rate"], args.lam_alarm)},
        },
        "alarm_cap": alarm_cap,
        "final_policy": {**final_stats, "utility": float(final_util)},
        "best_policy_checkpoint": None if best_ckpt is None else {"best_episode": best_ckpt.get("best_episode"), "best_utility": best_ckpt.get("best_utility")},
    }
    with open("results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Saved: results_summary.json")
    print("Saved: ddqn_model_last.pt")
    if best_ckpt is not None:
        print("Saved: ddqn_model_best.pt")

if __name__ == "__main__":
    main()
