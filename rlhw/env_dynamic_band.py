#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
env_dynamic_band.py (FINAL v4)

동적 Sigma Band 제어 (20분 그룹 단위)

핵심 아이디어
- 그룹(20분)별로 회귀선(yhat) + sigma를 계산하고,
- 밴드 계수 k를 "그룹 상황(관측)"에 따라 동적으로 선택한다.
- 목표: 고정 k 운전 대비
  (1) 불량(1) 검출 성능(posF1/TPR)을 높이면서,
  (2) 알람 비율(alarm_rate)을 과도하게 늘리지 않도록 trade-off를 학습한다.

⚠️ 중요(데이터 누설 방지)
- 관측(state)에는 라벨(PRED)을 절대 사용하지 않는다.
- 관측은 VALUE 기반 통계/추세만 사용한다.

데이터 스키마(기본)
- EQUIPMENTID (문자)
- GROUPNUM    (정수/문자)
- VALUE       (실수)
- PRED        (0/1 라벨, 불량=1)

split_meta.json이 있으면 그 안의 cols 매핑을 우선 사용한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    raise ImportError("gymnasium 필요: pip install gymnasium") from e


@dataclass(frozen=True)
class ActionSpec:
    k: float


def _detect_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise ValueError(f"Cannot find column among {candidates}. existing={list(df.columns)}")


def make_action_table(k_min: float, k_max: float, k_step: float) -> List[ActionSpec]:
    ks = np.arange(k_min, k_max + 1e-9, k_step)
    ks = [float(np.round(k, 3)) for k in ks]
    return [ActionSpec(k=k) for k in ks]


def pos_f1_from_counts(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    return float((2 * tp) / max(1, 2 * tp + fp + fn))


def micro_acc_from_counts(tp: int, tn: int, total: int) -> float:
    return float((tp + tn) / max(1, total))


class DynamicBandEnv(gym.Env):
    """
    Step = group(20분) 1개.
    reset() -> 첫 그룹 관측(state)
    step(action=k_index) -> 현재 그룹에 k 적용, reward 계산 후 다음 그룹 관측 반환.

    관측(state): 현재 그룹의 VALUE 기반 특징 + 이전 액션/알람률
      [sigma, diff_std, range_norm, slope_norm, prev_alarm_rate, prev_k_norm]
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        cols_map: Optional[Dict[str, str]] = None,

        # episode sampling
        episode_len: int = 60,
        random_start: bool = True,
        seed: int = 42,

        # band/model
        poly_deg: int = 3,
        sigma_mode: str = "value",   # "value" or "resid"
        sigma_floor: float = 0.01,

        # reward weights (count-based, 안정적)
        fp_pen: float = 0.30,
        fn_pen: float = 1.00,
        alarm_pen: float = 0.03,
        switch_cost: float = 0.02,

        # spike suppression (운영 안정성)
        alarm_cap: Optional[float] = None,
        spike_coef: float = 0.0,

        # action grid
        k_min: float = 2.0,
        k_max: float = 7.0,
        k_step: float = 0.5,

        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)

        # columns
        if cols_map is None:
            self.col_e = _detect_col(df, ["EQUIPMENTID", "equipment", "EID"])
            self.col_g = _detect_col(df, ["GROUPNUM", "group", "gid"])
            self.col_y = _detect_col(df, ["VALUE", "val", "x"])
            self.col_label = _detect_col(df, ["PRED", "label", "y", "target"])
        else:
            self.col_e = cols_map["EQUIPMENTID"]
            self.col_g = cols_map["GROUPNUM"]
            self.col_y = cols_map["VALUE"]
            self.col_label = cols_map["PRED"]

        self.df = df
        self.poly_deg = int(poly_deg)
        self.sigma_mode = str(sigma_mode)
        self.sigma_floor = float(sigma_floor)

        self.fp_pen = float(fp_pen)
        self.fn_pen = float(fn_pen)
        self.alarm_pen = float(alarm_pen)
        self.switch_cost = float(switch_cost)

        self.alarm_cap = None if alarm_cap is None else float(alarm_cap)
        self.spike_coef = float(spike_coef)

        self.episode_len = int(episode_len)
        self.random_start = bool(random_start)

        # actions
        self.action_table = make_action_table(k_min, k_max, k_step)
        self.action_space = spaces.Discrete(len(self.action_table))

        # obs: 6 dims
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # group map (key = (EQUIPMENTID, GROUPNUM))
        self.group_keys: List[Tuple[Any, Any]] = []
        self.group_data: Dict[Tuple[Any, Any], Dict[str, Any]] = {}

        self._prepare_groups()

        # episode state
        self._ep_keys: List[Tuple[Any, Any]] = []
        self._idx: int = 0
        self._prev_action: int = 0
        self._prev_alarm_rate: float = 0.0

        self.render_mode = render_mode

    def _prepare_groups(self):
        self.group_keys = []
        self.group_data = {}
        gb = self.df.groupby([self.col_e, self.col_g], sort=False)

        for key, g in gb:
            y = g[self.col_y].to_numpy(dtype=np.float64)
            y_true = g[self.col_label].to_numpy(dtype=int)

            n = int(len(y))
            x = np.arange(n, dtype=np.float64)

            # poly fit -> yhat
            if n >= self.poly_deg + 1:
                try:
                    coeffs = np.polyfit(x, y, self.poly_deg)
                    yhat = np.poly1d(coeffs)(x)
                except Exception:
                    yhat = np.full_like(y, float(np.mean(y)))
            else:
                yhat = np.full_like(y, float(np.mean(y)))

            resid = y - yhat
            sigma = float(np.std(y if self.sigma_mode == "value" else resid))
            if (not np.isfinite(sigma)) or sigma < self.sigma_floor:
                sigma = float(self.sigma_floor)

            # label-free features
            dy = np.diff(y) if n > 1 else np.array([0.0])
            diff_std = float(np.std(dy))
            y_min, y_max = float(np.min(y)), float(np.max(y))
            rng = y_max - y_min
            range_norm = float(rng / (sigma + 1e-9))

            # slope (linear) normalized
            if n > 1:
                try:
                    slope = float(np.polyfit(x, y, 1)[0])
                except Exception:
                    slope = 0.0
            else:
                slope = 0.0
            slope_norm = float(slope / (sigma + 1e-9))

            self.group_keys.append(key)
            self.group_data[key] = dict(
                y=y,
                y_true=y_true,
                yhat=yhat,
                sigma=sigma,
                feat=np.array([sigma, diff_std, range_norm, slope_norm], dtype=np.float32),
                n=n,
            )

    def reset(self, seed=None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        options = options or {}
        start_index = options.get("start_index", None)

        n_groups = len(self.group_keys)
        ep_len = min(self.episode_len, n_groups)

        if start_index is not None:
            s = int(start_index)
            s = max(0, min(s, max(0, n_groups - ep_len)))
        else:
            if not self.random_start:
                s = 0
            else:
                max_s = max(0, n_groups - ep_len)
                s = int(self.rng.integers(0, max_s + 1))

        self._ep_keys = self.group_keys[s : s + ep_len]
        self._idx = 0
        self._prev_action = 0
        self._prev_alarm_rate = 0.0

        obs = self._make_obs(self._ep_keys[self._idx])
        info = {"start_index": s, "group_key": self._ep_keys[self._idx]}
        return obs, info

    def _make_obs(self, key) -> np.ndarray:
        base = self.group_data[key]["feat"]
        prev_k_norm = float(self._prev_action) / float(max(1, len(self.action_table) - 1))
        obs = np.concatenate([base, np.array([self._prev_alarm_rate, prev_k_norm], dtype=np.float32)], axis=0)
        return obs.astype(np.float32)

    def step(self, action: int):
        action = int(action)
        spec = self.action_table[action]
        key = self._ep_keys[self._idx]
        gd = self.group_data[key]

        y = gd["y"]
        y_true = gd["y_true"]
        yhat = gd["yhat"]
        sigma = float(gd["sigma"])
        n = int(gd["n"])

        upper = yhat + spec.k * sigma
        lower = yhat - spec.k * sigma
        y_pred = ((y < lower) | (y > upper)).astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        alarm_points = int(np.sum(y_pred))
        alarm_rate = float(alarm_points / max(1, n))

        # reward (count-based, stable)
        reward = (tp - self.fp_pen * fp - self.fn_pen * fn) / max(1, n)
        reward -= self.alarm_pen * alarm_rate

        # switch cost
        if self._idx > 0 and action != self._prev_action:
            reward -= self.switch_cost

        # spike penalty (운영 안정성)
        if self.alarm_cap is not None and self.spike_coef > 0.0:
            over = max(0.0, alarm_rate - self.alarm_cap)
            if over > 0:
                reward -= float(self.spike_coef * (over ** 2))

        posf1 = pos_f1_from_counts(tp, fp, fn)
        micro = micro_acc_from_counts(tp, tn, n)

        info = dict(
            group_key=key,
            k=float(spec.k),
            n=n,
            sigma=float(sigma),
            tp=tp, fp=fp, fn=fn, tn=tn,
            posF1=float(posf1),
            micro_acc=float(micro),
            alarm_points=alarm_points,
            alarm_rate=float(alarm_rate),
        )

        # advance
        self._prev_action = action
        self._prev_alarm_rate = alarm_rate
        self._idx += 1

        terminated = self._idx >= len(self._ep_keys)
        truncated = False

        if not terminated:
            obs = self._make_obs(self._ep_keys[self._idx])
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, float(reward), terminated, truncated, info
