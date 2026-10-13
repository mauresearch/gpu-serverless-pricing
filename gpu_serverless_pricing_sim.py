"""Discrete-event simulator for game-theoretic pricing of multi-tenant
GPU-enabled serverless AI inference.

Layers
  1. Stackelberg pricing: the provider sets a unit price per model and round;
     users respond with a quasi-linear demand.
  2. Cooperative cost sharing: the boot and keep-alive overhead of each model
     and round is divided by the exact (closed-form) Shapley value of a
     keep-warm cost game and billed as the fixed part of a two-part tariff.
  3. SLO-aware dispatching: deadline-aware batching, admission control,
     fallback to a smaller model and overflow-aware scaling.

Times are in seconds and costs in US dollars.

Usage
    python gpu_serverless_pricing_sim.py --quick
    python gpu_serverless_pricing_sim.py --only all --out results_d01
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict, replace
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import simpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pickle

try:
    from scipy import stats as sps
except Exception:
    sps = None


# Default settings, used when the script is run without command-line arguments
# (e.g. from an IDE). Command-line options override them.
USER_SETTINGS = {
    "QUICK": False,                  # short smoke test
    "WORKLOAD": "azure_functions",   # "azure_functions" or "synthetic"
    "AZURE_FUNCTIONS_FILE": "data/invocations_per_function_md.anon.d01.csv",
    "AZURE_LLM_FILE": "data/AzureLLMInferenceTrace_conv.csv",
    "SEEDS": 10,                     # seeds for the main comparison and the ablation
    "SENS_SEEDS": 10,                # seeds per point of the sensitivity analysis
    "HORIZON_HOURS": 6.0,
    "USERS": 40,
    "WORKERS": 0,                    # 0 = all cores (1 inside an IDE console)
    "ONLY": "all",                   # all | core | main | ablation | sensitivity
    "OUT_DIR": "results",
    "REPLOT": False,                 # redraw figures from results/raw_results.pkl
}


def _in_spyder():
    return any(k.startswith("SPYDER") for k in os.environ) or "spyder_kernels" in sys.modules


@dataclass(frozen=True)
class ModelSpec:
    name: str
    gpu: str
    gpu_price_h: float
    cold_start_s: float
    t_iso_s: float
    kappa: float
    slo_factor: float
    slo_floor_s: float
    max_instances: int
    user_share: float
    token_based: bool


# Parameter sources: GPU prices from the Modal price list; cold starts from
# Torpor (ATC'25, ResNet-152 8 s, Llama3-8B 48 s) and ServerlessLLM (OSDI'24,
# LLaMA-2-70B on 8 GPUs 84 s); latencies from Torpor and Splitwise (ISCA'24);
# slowdown SLO of 5x isolated latency from Splitwise. The mean request of the
# Azure LLM conversation trace has 1155 prompt and 211 output tokens.
DEFAULT_MODELS = (
    ModelSpec("S-Vision", "L4", 0.80, 8.0, 0.011, 8.0, 5.0, 0.150, 4, 0.35, False),
    ModelSpec("M-LLM-7B", "A100-80GB", 2.50, 48.0, 0.104 + 211 * 0.014, 50.0, 5.0, 0.0, 4, 0.45, True),
    ModelSpec("L-LLM-70B", "8xH100", 8 * 3.95, 84.0, 0.084 * 1155 / 1020 + 211 * 0.028, 50.0, 5.0, 0.0,
              3, 0.20, True),
)


@dataclass
class Config:
    horizon_h: float = 6.0
    round_min: float = 15.0
    drain_s: float = 120.0
    seed: int = 1
    pop_seed: int = 1
    n_users: int = 40
    models: tuple = DEFAULT_MODELS
    rate_mean_per_min: float = 4.0
    rate_sigma: float = 1.2
    cv_median: float = 1.5
    cv_sigma: float = 0.6
    diurnal_amp: float = 0.3
    load_scale: float = 1.0
    workload: str = "synthetic"
    azure_functions_path: str = ""
    azure_llm_path: str = ""
    keepalive_s: float = 60.0
    cold_scale: float = 1.0
    sla_scale: float = 1.0
    markup_ref: float = 1.0
    ref_utilization: float = 0.15
    demand_elasticity: float = 1.2
    refund_frac: float = 0.30
    cs_weight: float = 0.5
    est_ema: float = 0.5
    pricing: str = "stackelberg"
    allocation: str = "shapley"
    batching: str = "sla_aware"
    tariff: str = "two_part"
    admission: bool = False
    admission_slack: float = 0.0
    admission_strict: bool = False
    prewarm: bool = False
    prewarm_util: float = 0.6
    keepalive_policy: str = "fixed"
    batch_max_wait_frac: float = 1.0
    fallback: bool = False
    fallback_value: float = 0.7
    late_value: float = 0.5
    overflow_frac: float = 0.5
    fixed_batch: int = 8
    fixed_timeout_frac: float = 0.25
    sla_margin_frac: float = 0.10
    target_util: float = 0.0
    batch_cap_frac: float = 0.5
    shapley_method: str = "closed_form"
    shapley_perms: int = 64
    shapley_exact_max: int = 8
    core_samples: int = 300
    core_exact_max: int = 8
    keep_details: bool = False
    calibrate: bool = False


_TRACE_CACHE: dict = {}


def load_llm_token_works(path: str):
    if not path:
        return None
    key = ("llm", path)
    if key in _TRACE_CACHE:
        return _TRACE_CACHE[key]
    df = pd.read_csv(path)
    w = 0.005 * df["ContextTokens"].to_numpy(float) + df["GeneratedTokens"].to_numpy(float)
    w = w[np.isfinite(w) & (w > 0)]
    w = w / w.mean()
    w = np.clip(w, 0.02, 8.0)
    w = w / w.mean()
    ts = pd.to_datetime(df["TIMESTAMP"])
    sec = (ts - ts.min()).dt.total_seconds().to_numpy()
    out = {"work": w, "ctx": df["ContextTokens"].to_numpy(), "gen": df["GeneratedTokens"].to_numpy(),
           "arrival_s": np.sort(sec)}
    _TRACE_CACHE[key] = out
    return out


def load_azure_functions(path: str, n_users: int, minutes: int, seed: int):
    key = ("af", path, n_users, minutes, seed)
    if key in _TRACE_CACHE:
        return _TRACE_CACHE[key]
    df = pd.read_csv(path)
    cols = [str(i) for i in range(1, 1441)]
    mat = df[cols].to_numpy(float)
    start = 8 * 60
    win = mat[:, start:start + minutes] if start + minutes <= 1440 else mat[:, :minutes]
    tot = win.sum(1)
    ok = np.flatnonzero((tot >= minutes / 10.0) & (tot <= minutes * 60.0))
    rng = np.random.default_rng([seed, 99])
    pick = rng.choice(ok, size=min(n_users, ok.size), replace=False)
    out = win[pick]
    _TRACE_CACHE[key] = out
    return out


@dataclass
class User:
    uid: int
    m: int
    q0_rate: float
    cv: float
    pattern: np.ndarray | None = None


def make_users(cfg: Config):
    rng = np.random.default_rng([cfg.pop_seed, 7])
    M = len(cfg.models)
    shares = np.array([s.user_share for s in cfg.models], float)
    shares /= shares.sum()
    ms = list(np.repeat(np.arange(M), 2))
    ms += list(rng.choice(M, size=cfg.n_users - len(ms), p=shares))
    rng.shuffle(ms)
    mean = cfg.rate_mean_per_min / 60.0
    mu = math.log(mean) - cfg.rate_sigma ** 2 / 2
    rates = rng.lognormal(mu, cfg.rate_sigma, cfg.n_users)
    cvs = np.clip(rng.lognormal(math.log(cfg.cv_median), cfg.cv_sigma, cfg.n_users), 0.3, 6.0)
    users = [User(i, int(ms[i]), float(rates[i]), float(cvs[i])) for i in range(cfg.n_users)]
    if cfg.workload == "azure_functions" and cfg.azure_functions_path:
        minutes = int(math.ceil(cfg.horizon_h * 60))
        pat = load_azure_functions(cfg.azure_functions_path, cfg.n_users, minutes, cfg.pop_seed)
        scale = mean * 60.0 / max(1e-9, pat.mean())
        for i, u in enumerate(users[:pat.shape[0]]):
            u.pattern = pat[i] * scale
            u.q0_rate = float(u.pattern.mean() / 60.0)
            u.cv = float("nan")
    return users


def diurnal(cfg: Config, t: float):
    return 1.0 + cfg.diurnal_amp * math.sin(2 * math.pi * (t / 86400.0 + 8 / 24 - 0.25))


def base_demand_round(cfg: Config, u: User, r: int, T: float):
    if u.pattern is not None:
        a, b = int(r * T / 60), int((r + 1) * T / 60)
        return float(u.pattern[a:b].sum()) * cfg.load_scale
    return u.q0_rate * T * diurnal(cfg, (r + 0.5) * T) * cfg.load_scale


def gamma_arrivals(rng, q: float, cv: float, T: float):
    if q <= 1e-9:
        return np.empty(0)
    k = 1.0 / cv ** 2
    scale = (T / q) / k
    n = int(q + 6 * math.sqrt(q) * max(1.0, cv) + 10)
    gaps = rng.gamma(k, scale, n)
    gaps[0] *= rng.random()
    t = np.cumsum(gaps)
    while t[-1] < T:
        more = np.cumsum(rng.gamma(k, scale, n)) + t[-1]
        t = np.concatenate([t, more])
    return t[t < T]


def trace_arrivals(rng, pattern: np.ndarray, r: int, T: float, factor: float):
    a, b = int(r * T / 60), int((r + 1) * T / 60)
    out = []
    for j, c in enumerate(pattern[a:b]):
        n = rng.poisson(c * factor)
        if n:
            out.append(j * 60.0 + rng.random(n) * 60.0)
    return np.sort(np.concatenate(out)) if out else np.empty(0)


def sample_works(cfg: Config, rng, spec: ModelSpec, n: int):
    if not spec.token_based:
        return np.ones(n)
    tr = load_llm_token_works(cfg.azure_llm_path)
    if tr is not None:
        return rng.choice(tr["work"], size=n)
    s = 0.7
    return rng.lognormal(-s * s / 2, s, n)


def user_intervals(times: np.ndarray, ka: float):
    if times.size == 0:
        return np.empty(0), np.empty(0)
    t = np.sort(times)
    brk = np.flatnonzero(np.diff(t) > ka)
    starts = np.concatenate([[t[0]], t[brk + 1]])
    ends = np.concatenate([t[brk], [t[-1]]]) + ka
    return starts, ends


def merge_cost(starts, ends, boot_cost, idle_price):
    if starts.size == 0:
        return 0.0, starts, ends
    o = np.argsort(starts, kind="stable")
    s, e = starts[o], ends[o]
    cm = np.maximum.accumulate(e)
    new = np.empty(s.size, bool)
    new[0] = True
    new[1:] = s[1:] > cm[:-1]
    idx = np.flatnonzero(new)
    ce = np.append(cm[idx[1:] - 1], cm[-1])
    cs = s[idx]
    return boot_cost * idx.size + idle_price * float((ce - cs).sum()), cs, ce


class CostGame:
    def __init__(self, times_by_user: dict, ka: float, boot_cost: float, idle_price: float):
        self.players = sorted(times_by_user)
        self.iv = {u: user_intervals(np.asarray(times_by_user[u]), ka) for u in self.players}
        self.bc, self.ip = boot_cost, idle_price

    def cost(self, S):
        if len(S) == 0:
            return 0.0
        st = np.concatenate([self.iv[u][0] for u in S])
        en = np.concatenate([self.iv[u][1] for u in S])
        return merge_cost(st, en, self.bc, self.ip)[0]

    def shapley_closed_form(self):
        """Exact Shapley value of c(S) = boot_cost * B(S) + idle_price * U(S).

        U(S) is the length of the union of warm intervals: each elementary
        segment is split equally among the users covering it. B(S) is the
        number of connected components: the start of an interval owned by u
        and covered by k other users gives 1/(k+1) to u and -1/(k(k+1)) to
        each covering user. Runs in O(n_I^2) for n_I warm intervals.
        """
        P = self.players
        if not P:
            return {}
        st = np.concatenate([self.iv[u][0] for u in P])
        en = np.concatenate([self.iv[u][1] for u in P])
        us = np.concatenate([np.full(self.iv[u][0].size, k) for k, u in enumerate(P)])
        n = len(P)
        phiB = np.zeros(n)
        for j in range(st.size):
            cov = (st <= st[j]) & (en >= st[j]) & (us != us[j])
            covu = np.unique(us[cov])
            k = covu.size
            phiB[us[j]] += 1.0 / (k + 1)
            if k:
                phiB[covu] -= 1.0 / (k * (k + 1))
        pts = np.unique(np.concatenate([st, en]))
        phiU = np.zeros(n)
        for a, b in zip(pts[:-1], pts[1:]):
            mid = 0.5 * (a + b)
            covu = np.unique(us[(st <= mid) & (en >= mid)])
            if covu.size:
                phiU[covu] += (b - a) / covu.size
        phi = self.bc * phiB + self.ip * phiU
        return {u: float(phi[k]) for k, u in enumerate(P)}

    def shapley(self, rng, perms: int, exact_max: int, method: str = "mc"):
        if method == "closed_form":
            return self.shapley_closed_form()
        P = self.players
        n = len(P)
        if n == 0:
            return {}
        if n == 1:
            return {P[0]: self.cost(P)}
        if n <= exact_max:
            cost = np.zeros(1 << n)
            for mask in range(1, 1 << n):
                cost[mask] = self.cost([P[j] for j in range(n) if mask >> j & 1])
            fact = [math.factorial(k) for k in range(n + 1)]
            phi = np.zeros(n)
            for mask in range(1 << n):
                k = bin(mask).count("1")
                for j in range(n):
                    if not mask >> j & 1:
                        w = fact[k] * fact[n - k - 1] / fact[n]
                        phi[j] += w * (cost[mask | (1 << j)] - cost[mask])
            return dict(zip(P, phi))
        phi = defaultdict(float)
        n_pairs = max(1, perms // 2)
        for _ in range(n_pairs):
            perm = list(rng.permutation(n))
            for order in (perm, perm[::-1]):
                cs, ce = np.empty(0), np.empty(0)
                prev = 0.0
                for j in order:
                    u = P[j]
                    c, cs, ce = merge_cost(np.concatenate([cs, self.iv[u][0]]),
                                           np.concatenate([ce, self.iv[u][1]]), self.bc, self.ip)
                    phi[u] += c - prev
                    prev = c
        return {u: phi[u] / (2 * n_pairs) for u in P}


class Req:
    __slots__ = ("rid", "user", "m", "rnd", "arr", "work", "cold", "om", "slo")

    def __init__(self, rid, user, m, rnd, arr, work):
        self.rid, self.user, self.m, self.rnd, self.arr, self.work = rid, user, m, rnd, arr, work
        self.cold = False
        self.om = m
        self.slo = float("inf")


class Instance:
    __slots__ = ("state", "alloc_start", "boot_end", "idle_since", "epoch", "last_user",
                 "busy_until")

    def __init__(self):
        self.state = "booting"
        self.epoch = 0
        self.busy_until = 0.0


class Logs:
    def __init__(self):
        self.req = []
        self.arrivals = []
        self.boots = []
        self.idle = []
        self.batches = []
        self.rejects = []


class ModelServer:
    def __init__(self, env, m, spec: ModelSpec, cfg: Config, logs: Logs, T: float, R: int):
        self.env, self.m, self.spec, self.cfg, self.logs = env, m, spec, cfg, logs
        self.T, self.R = T, R
        self.cold_s = spec.cold_start_s * cfg.cold_scale
        self.T1 = spec.t_iso_s
        self.kappa = spec.kappa
        self.sla = self.req_slo(1.0)
        self.price_s = spec.gpu_price_h / 3600.0
        self.queue: list[Req] = []
        self.instances: list[Instance] = []
        self.wake = env.event()
        self.gap_ema = None
        self.last_arr = None
        self.gaps = []
        if cfg.batching == "none":
            self.B = 1
        elif cfg.batching == "fixed":
            self.B = cfg.fixed_batch
        else:
            b = int(1 + self.kappa * (cfg.batch_cap_frac * self.sla / self.T1 - 1))
            self.B = int(np.clip(b, 1, 32))
        self.w95 = float(np.quantile(sample_works(cfg, np.random.default_rng(0), spec, 4000), 0.95))
        self.value_per_req = 0.0
        self.fallback_server = None
        self.overflow = []
        env.process(self.dispatcher())

    def rnd(self, t):
        return min(self.R - 1, int(t // self.T))

    def kick(self):
        if not self.wake.triggered:
            self.wake.succeed()

    def submit(self, req: Req):
        now = self.env.now
        if self.last_arr is not None:
            g = now - self.last_arr
            self.gap_ema = g if self.gap_ema is None else 0.9 * self.gap_ema + 0.1 * g
            self.gaps.append(g)
            if len(self.gaps) > 3000:
                del self.gaps[:1000]
        self.last_arr = now
        if self.fallback_server is not None and not self.admit(req, now, strict=True):
            fb = self.fallback_server
            if fb.admit(req, now, strict=True):
                self.pressure(req, now)
                req.m = fb.m
                fb.enqueue(req, now)
                return
        if self.cfg.admission and not self.admit(req, now):
            self.logs.rejects.append((req.user, self.m, req.rnd, now))
            self.pressure(req, now)
            return
        self.enqueue(req, now)

    def pressure(self, req, now):
        self.overflow.append(now)
        while self.overflow and self.overflow[0] < now - self.cold_s:
            self.overflow.pop(0)
        if len(self.overflow) >= max(1, int(self.cfg.overflow_frac * self.B)) and \
                len(self.instances) < self.spec.max_instances and \
                not any(i.state == "booting" for i in self.instances):
            self.overflow.clear()
            self.boot(req.user, now)

    def enqueue(self, req, now):
        self.queue.append(req)
        self.logs.arrivals.append((req.user, self.m, req.rnd, now))
        self.kick()

    def admit(self, req, now, strict=None):
        """Admission test: predicted completion time vs. (1 + slack) * SLO."""
        warm = [i for i in self.instances if i.state in ("idle", "busy")]
        booting = [i for i in self.instances if i.state == "booting"]
        strict = self.cfg.admission_strict if strict is None else strict
        deadline = req.arr + req.slo * (1.0 + self.cfg.admission_slack)
        own = self.T1 * req.work
        if not warm:
            if any(i.boot_end + own <= deadline for i in booting):
                return True
            if now + self.cold_s + own <= deadline:
                return True
            return not strict
        n = len(warm)
        ahead = len(self.queue)
        t_free = min((max(i.busy_until, now) if i.state == "busy" else now) for i in warm)
        rounds = ahead // (n * self.B)
        rest = ahead % (n * self.B)
        est = t_free + rounds * self.T1 * self.w95 * self.f(self.B) + own * self.f(min(self.B, rest + 1))
        if est <= deadline:
            return True
        return any(i.boot_end + own <= deadline for i in booting)

    def req_slo(self, w):
        return max(self.spec.slo_factor * self.T1 * w, self.spec.slo_floor_s) * self.cfg.sla_scale

    def warm_need(self):
        if not self.cfg.prewarm or self.gap_ema is None or self.gap_ema <= 0:
            return 0
        gap = max(self.gap_ema, self.env.now - self.last_arr) if self.last_arr is not None else self.gap_ema
        lam = 1.0 / gap
        thr = self.B / (self.T1 * self.f(self.B))
        return int(min(self.spec.max_instances, math.ceil(lam / (thr * self.cfg.prewarm_util))))

    def dispatcher(self):
        while True:
            t_next = self.try_dispatch()
            if t_next is None:
                yield self.wake
            else:
                yield self.wake | self.env.timeout(max(0.0, t_next - self.env.now))
            if self.wake.triggered:
                self.wake = self.env.event()

    def f(self, b):
        return 1.0 + (max(int(b), 1) - 1) / self.kappa

    def target_batch(self):
        if self.gap_ema is None or self.gap_ema <= 0:
            return 1
        n_act = max(1, sum(1 for i in self.instances if i.state != "dead"))
        lam = (1.0 / self.gap_ema) / n_act / self.cfg.target_util
        for b in range(1, self.B + 1):
            if b / (self.T1 * self.f(b)) >= lam:
                return b
        return self.B

    def batch_decision(self, now):
        """Return (dispatch_now, next_check_time) for the head of the queue."""
        q = self.queue
        n = len(q)
        if self.cfg.batching == "none" or n >= self.B:
            return True, None
        if self.cfg.batching == "sla_aware" and self.cfg.target_util > 0 and n >= self.target_batch():
            return True, None
        if self.cfg.batching == "fixed":
            t_dead = q[0].arr + self.cfg.fixed_timeout_frac * q[0].slo
            return (now >= t_dead), t_dead
        fn = self.f(min(n + 1, self.B))
        latest_start = min(r.arr + r.slo * (1 - self.cfg.sla_margin_frac) - self.T1 * r.work * fn
                           for r in q[:self.B])
        latest_start = min(latest_start, q[0].arr + self.cfg.batch_max_wait_frac * q[0].slo)
        if now >= latest_start:
            return True, None
        exp_gap = self.gap_ema if self.gap_ema is not None else float("inf")
        if now + exp_gap > latest_start:
            return True, None
        return False, latest_start

    def try_dispatch(self):
        now = self.env.now
        while self.queue:
            idle = next((i for i in self.instances if i.state == "idle"), None)
            if idle is None:
                self.maybe_scale_up(now)
                return None
            go, t_next = self.batch_decision(now)
            if not go:
                return t_next
            batch = self.queue[:self.B]
            del self.queue[:self.B]
            self.start_batch(idle, batch, now)
        return None

    def maybe_scale_up(self, now):
        if len(self.instances) >= self.spec.max_instances:
            return
        n_boot = sum(1 for i in self.instances if i.state == "booting")
        n_soon = sum(1 for i in self.instances
                     if i.state == "busy" and i.busy_until <= now + self.cold_s)
        if len(self.queue) > (n_boot + n_soon) * self.B:
            trig = self.queue[min(len(self.queue) - 1, (n_boot + n_soon) * self.B)]
            self.boot(trig.user, now)

    def boot(self, trigger_user, now):
        inst = Instance()
        inst.alloc_start = now
        inst.boot_end = now + self.cold_s
        inst.last_user = trigger_user
        self.instances.append(inst)
        self.logs.boots.append((self.m, now, self.cold_s * self.price_s, trigger_user, self.rnd(now)))
        self.env.process(self._boot_proc(inst))

    def _boot_proc(self, inst):
        yield self.env.timeout(self.cold_s)
        self.set_idle(inst)
        self.kick()

    def set_idle(self, inst):
        inst.state = "idle"
        inst.idle_since = self.env.now
        inst.epoch += 1
        self.env.process(self._keepalive(inst, inst.epoch))

    def set_idle_keep(self, inst):
        inst.epoch += 1
        self.env.process(self._keepalive(inst, inst.epoch))

    def close_idle(self, inst, t_end):
        dt = t_end - inst.idle_since
        if dt > 0:
            self.logs.idle.append((self.m, inst.idle_since, t_end, dt * self.price_s,
                                   inst.last_user, self.rnd(inst.idle_since)))

    def ka_window(self):
        if self.cfg.keepalive_policy != "adaptive":
            return self.cfg.keepalive_s
        others = sum(1 for i in self.instances if i.state in ("idle", "busy", "booting")) - 1
        if others > 0:
            return max(self.cold_s, 10.0)
        if len(self.gaps) < 30:
            return self.cfg.keepalive_s
        G = np.asarray(self.gaps[-1500:])
        lam = 1.0 / max(1e-6, G.mean())
        boot_cost = self.cold_s * self.price_s
        sla_loss = self.value_per_req * (1.0 + lam * self.cold_s)
        W = np.array([5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 450, 600, 900], float)
        W = W[W >= min(10.0, self.cold_s)]
        exp_idle = np.array([np.minimum(G, w).mean() for w in W]) * self.price_s
        p_miss = np.array([(G > w).mean() for w in W])
        cost = exp_idle + p_miss * (boot_cost + sla_loss)
        return float(W[int(np.argmin(cost))])

    def _keepalive(self, inst, ep):
        yield self.env.timeout(self.ka_window())
        if inst.state == "idle" and inst.epoch == ep:
            live = sum(1 for i in self.instances if i.state != "dead")
            if live <= self.warm_need() and self.env.now < self.T * self.R:
                self.set_idle_keep(inst)
                return
            self.close_idle(inst, self.env.now)
            inst.state = "dead"
            self.instances.remove(inst)

    def start_batch(self, inst, batch, now):
        self.close_idle(inst, now)
        inst.state = "busy"
        for r in batch:
            r.cold = inst.boot_end > r.arr
        fb = self.f(len(batch))
        durs = [self.T1 * r.work * fb for r in batch]
        dur = max(durs)
        inst.busy_until = now + dur
        self.logs.batches.append((self.m, now, dur, len(batch), dur * self.price_s, self.rnd(now)))
        self.env.process(self._exec(inst, batch, now, dur, durs))

    def _exec(self, inst, batch, t0, dur, durs):
        yield self.env.timeout(dur)
        for r, d in zip(batch, durs):
            self.logs.req.append((r.rid, r.user, self.m, r.rnd, r.arr, t0, t0 + d, len(batch), r.cold,
                                  r.om, r.slo))
        inst.last_user = batch[-1].user
        self.set_idle(inst)
        self.kick()

    def finalize(self, t_end):
        for inst in self.instances:
            if inst.state == "idle":
                self.close_idle(inst, min(t_end, inst.idle_since + self.cfg.keepalive_s))


def c_exec(spec: ModelSpec, b: float):
    return spec.gpu_price_h / 3600.0 * spec.t_iso_s * (1.0 + (max(b, 1) - 1) / spec.kappa) / b


def demand(q0, p, e, p_ref, eps):
    return np.maximum(0.0, q0 * (1.0 - eps * (p + e - p_ref) / p_ref))


def utility(q, q0, pay, p_ref, eps):
    a = p_ref * (1.0 + 1.0 / eps)
    if q0 <= 1e-12:
        return -pay
    b = p_ref / (eps * q0)
    return a * q - 0.5 * b * q * q - pay


def follower_response(q0, p, e, fee, p_ref, eps):
    """Best response under a linear (e > 0) or two-part (fee > 0) tariff."""
    q = demand(q0, p, e, p_ref, eps)
    if eps <= 0:
        return q, np.zeros_like(q)
    a = p_ref * (1 + 1 / eps)
    cs = 0.5 * np.clip(a - p - e, 0, None) * q
    active = cs >= fee
    return q * active, (cs - fee) * active


def leader_objective(p, q0h, eh, p_ref, eps, ch, vh, refund, cs_w, fee=None):
    p = np.atleast_1d(p)[:, None]
    fee = np.zeros_like(q0h) if fee is None else fee
    q, ncs = follower_response(q0h[None, :], p, eh[None, :], fee[None, :], p_ref, eps)
    prof = ((p * (1 - refund * vh) - ch) * q).sum(1)
    if cs_w:
        prof = prof + cs_w * ncs.sum(1)
    return prof


def solve_stackelberg(q0h, eh, p_ref, eps, ch, vh, refund, cs_w, fee=None):
    """Leader price by backward induction: grid search + golden-section refinement."""
    if q0h.size == 0 or q0h.sum() <= 0:
        return 2.0 * max(ch, 1e-12)
    lo = max(ch, 1e-12)
    hi = p_ref * (1 + 1 / eps) * 1.5
    grid = np.linspace(lo, hi, 400)
    val = leader_objective(grid, q0h, eh, p_ref, eps, ch, vh, refund, cs_w, fee)
    k = int(np.argmax(val))
    a, b = grid[max(0, k - 1)], grid[min(len(grid) - 1, k + 1)]
    g = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        c1, c2 = b - g * (b - a), a + g * (b - a)
        f1, f2 = (leader_objective(np.array([c]), q0h, eh, p_ref, eps, ch, vh, refund, cs_w, fee)[0]
                  for c in (c1, c2))
        if f1 > f2:
            b = c2
        else:
            a = c1
    return 0.5 * (a + b)


def allocate_cell(method, times_by_user, O, boots_cell, idle_cell, cfg, spec, rng, shap_cache, key):
    """Allocate the measured overhead O of one (round, model) among its users."""
    users = sorted(times_by_user)
    if O <= 0:
        return {}
    if method == "trigger" or not users:
        x = defaultdict(float)
        for (_, _, c, u, _) in boots_cell:
            x[u] += c
        for (_, _, _, c, u, _) in idle_cell:
            x[u] += c
        tot = sum(x.values())
        return {u: v * O / tot for u, v in x.items()} if tot > 0 else {}
    if method == "equal":
        return {u: O / len(users) for u in users}
    if method == "proportional":
        n = {u: len(times_by_user[u]) for u in users}
        N = sum(n.values())
        return {u: O * n[u] / N for u in users}
    if key not in shap_cache:
        g = CostGame(times_by_user, cfg.keepalive_s, spec.cold_start_s * cfg.cold_scale * spec.gpu_price_h / 3600,
                     spec.gpu_price_h / 3600)
        shap_cache[key] = g.shapley(rng, cfg.shapley_perms, cfg.shapley_exact_max, cfg.shapley_method)
    phi = shap_cache[key]
    tot = sum(max(0.0, v) for v in phi.values())
    if tot <= 0:
        return {u: O / len(users) for u in users}
    return {u: O * max(0.0, phi[u]) / tot for u in users}


def group_cell_data(logs: Logs, R: int, M: int):
    arr = defaultdict(lambda: defaultdict(list))
    for (u, m, r, t) in logs.arrivals:
        arr[(r, m)][u].append(t)
    boots = defaultdict(list)
    for b in logs.boots:
        boots[(b[4], b[0])].append(b)
    idle = defaultdict(list)
    for d in logs.idle:
        idle[(d[5], d[0])].append(d)
    return arr, boots, idle


_REF_CACHE: dict = {}


def reference_overhead(cfg: Config):
    """Pilot run of current practice; returns the overhead per request of each model."""
    key = (cfg.pop_seed, cfg.n_users, cfg.workload, cfg.azure_functions_path, cfg.azure_llm_path,
           cfg.rate_mean_per_min, cfg.rate_sigma, cfg.cv_median, cfg.cv_sigma, cfg.diurnal_amp, cfg.models)
    if key not in _REF_CACHE:
        base = Config()
        pc = replace(base, pop_seed=cfg.pop_seed, n_users=cfg.n_users, workload=cfg.workload,
                     azure_functions_path=cfg.azure_functions_path, azure_llm_path=cfg.azure_llm_path,
                     rate_mean_per_min=cfg.rate_mean_per_min, rate_sigma=cfg.rate_sigma,
                     cv_median=cfg.cv_median, cv_sigma=cfg.cv_sigma, diurnal_amp=cfg.diurnal_amp,
                     models=cfg.models, horizon_h=min(3.0, cfg.horizon_h * 2), seed=987654,
                     pricing="fixed", allocation="proportional", batching="fixed",
                     tariff="linear", demand_elasticity=0.0, calibrate=True)
        _REF_CACHE[key] = simulate(pc)
    return _REF_CACHE[key]


def simulate(cfg: Config) -> dict:
    T = cfg.round_min * 60.0
    R = int(round(cfg.horizon_h * 60 / cfg.round_min))
    H = R * T
    M = len(cfg.models)
    users = make_users(cfg)
    U = len(users)
    eps = cfg.demand_elasticity
    p_fix = np.array([(1 + cfg.markup_ref) * c_exec(s, 1.0) for s in cfg.models])
    if cfg.calibrate:
        e_ref = np.zeros(len(cfg.models))
    else:
        e_ref = reference_overhead(cfg)
    p_ref = p_fix + e_ref
    by_model = [[u.uid for u in users if u.m == m] for m in range(M)]

    env = simpy.Environment()
    logs = Logs()
    servers = [ModelServer(env, m, s, cfg, logs, T, R) for m, s in enumerate(cfg.models)]
    for m, sv in enumerate(servers):
        sv.value_per_req = float(p_ref[m]) * (1.0 + cfg.refund_frac)
        if cfg.fallback and m > 0 and cfg.models[m].token_based and cfg.models[m - 1].token_based:
            sv.fallback_server = servers[m - 1]

    q0_hat = np.array([base_demand_round(cfg, u, 0, T) for u in users])
    e_hat = np.array([e_ref[u.m] for u in users])
    f_hat = np.array([e_ref[u.m] * base_demand_round(cfg, u, 0, T) for u in users])
    two_part = cfg.tariff == "two_part"
    c_hat = np.array([c_exec(s, 1.0) for s in cfg.models])
    v_hat = np.zeros(M)
    r_hat = np.zeros(M)
    shap_cache: dict = {}
    rng_alloc = np.random.default_rng([cfg.seed, 31])

    prices = np.zeros((R, M))
    q_true0 = np.zeros((R, U))
    q_plan = np.zeros((R, U))
    e_used = np.zeros((R, U))
    f_used = np.zeros((R, U))

    def eff(ids):
        if two_part:
            return np.zeros(len(ids)), f_hat[ids]
        return e_hat[ids], np.zeros(len(ids))
    snapshot = {}
    rid_counter = [0]

    def injector(evts):
        for (t, uid, m, rnd, w) in evts:
            d = t - env.now
            if d > 0:
                yield env.timeout(d)
            rid_counter[0] += 1
            rq_ = Req(rid_counter[0], uid, m, rnd, env.now, w)
            rq_.slo = servers[m].req_slo(w)
            servers[m].submit(rq_)

    def market():
        for r in range(R):
            if r > 0:
                _update_beliefs(r - 1)
            for m in range(M):
                ids = np.array(by_model[m], int)
                if cfg.pricing == "fixed":
                    prices[r, m] = p_fix[m]
                else:
                    ee, ff = eff(ids)
                    v_eff = (1 - (1 - r_hat[m]) * (1 - cfg.refund_frac * v_hat[m])) / max(cfg.refund_frac, 1e-9)
                    prices[r, m] = solve_stackelberg(q0_hat[ids], ee, p_ref[m], eps, c_hat[m], v_eff,
                                                     cfg.refund_frac, cfg.cs_weight, ff)
            if r == R // 2:
                ee, ff = eff(np.arange(U))
                snapshot.update(q0_hat=q0_hat.copy(), e_hat=ee.copy(), f_hat=ff.copy(), c_hat=c_hat.copy(),
                                v_hat=v_hat.copy(), p=prices[r].copy(), p_ref=p_ref.copy(), p_fix=p_fix.copy(),
                                by_model=by_model)
            evts = []
            for u in users:
                q0 = base_demand_round(cfg, u, r, T)
                q_true0[r, u.uid] = q0
                ee, ff = eff(np.array([u.uid]))
                e_used[r, u.uid], f_used[r, u.uid] = ee[0], ff[0]
                q = float(follower_response(q0, prices[r, u.m], ee[0], ff[0], p_ref[u.m], eps)[0])
                q_plan[r, u.uid] = q
                rng = np.random.default_rng([cfg.seed, 1000 + u.uid, r])
                if u.pattern is not None:
                    ts = trace_arrivals(rng, u.pattern, r, T, cfg.load_scale * q / q0 if q0 > 0 else 0.0)
                else:
                    ts = gamma_arrivals(rng, q, u.cv, T)
                ws = sample_works(cfg, rng, cfg.models[u.m], ts.size)
                evts += [(r * T + t, u.uid, u.m, r, w) for t, w in zip(ts, ws)]
            evts.sort(key=lambda e: e[0])
            env.process(injector(evts))
            yield env.timeout(T)

    def _update_beliefs(rp):
        a = cfg.est_ema
        arr, boots, idle = group_cell_data(logs, R, M)
        n_obs = np.zeros(U)
        for m in range(M):
            for u, ts in arr[(rp, m)].items():
                n_obs[u] += len(ts)
        rej_m = np.zeros(M)
        for (u, m, rr, t) in logs.rejects:
            if rr == rp:
                n_obs[u] += 1
                rej_m[m] += 1
        for m in range(M):
            tot = sum(len(ts) for ts in arr[(rp, m)].values()) + rej_m[m]
            if tot > 0:
                r_hat[m] = (1 - a) * r_hat[m] + a * rej_m[m] / tot
        x_round = np.zeros(U)
        for m in range(M):
            O = sum(b[2] for b in boots[(rp, m)]) + sum(d[3] for d in idle[(rp, m)])
            x = allocate_cell(cfg.allocation, arr[(rp, m)], O, boots[(rp, m)], idle[(rp, m)],
                              cfg, cfg.models[m], rng_alloc, shap_cache, (rp, m))
            for u, v in x.items():
                x_round[u] += v
        for u in users:
            i = u.uid
            m = u.m
            factor = 1.0 - eps * (prices[rp, m] + e_used[rp, i] - p_ref[m]) / p_ref[m]
            if factor > 0.05 and n_obs[i] > 0:
                q0_hat[i] = (1 - a) * q0_hat[i] + a * n_obs[i] / factor
            if n_obs[i] > 0:
                e_hat[i] = (1 - a) * e_hat[i] + a * x_round[i] / n_obs[i]
                f_hat[i] = (1 - a) * f_hat[i] + a * x_round[i]
            else:
                f_hat[i] *= 0.9
                ids = by_model[m]
                tot_n = sum(n_obs[j] for j in ids)
                avg = sum(x_round[j] for j in ids) / tot_n if tot_n > 0 else e_hat[i]
                e_hat[i] = 0.5 * e_hat[i] + 0.5 * avg
        for m in range(M):
            bt = [b for b in logs.batches if b[0] == m and b[5] == rp]
            n = sum(b[3] for b in bt)
            if n > 0:
                c_hat[m] = (1 - a) * c_hat[m] + a * sum(b[4] for b in bt) / n
            fin = [(row[6] - row[4]) > row[10] for row in logs.req if row[9] == m and row[3] == rp]
            if fin:
                v_hat[m] = (1 - a) * v_hat[m] + a * float(np.mean(fin))

    env.process(market())
    env.run(until=H + cfg.drain_s)
    for s in servers:
        s.finalize(env.now)
    if cfg.calibrate:
        out = np.zeros(M)
        for m, s in enumerate(cfg.models):
            ov = sum(b[2] for b in logs.boots if b[0] == m) + sum(d[3] for d in logs.idle if d[0] == m)
            n = sum(1 for a_ in logs.arrivals if a_[1] == m)
            out[m] = ov / n if n > 0 else c_exec(s, 1.0) * (1 / cfg.ref_utilization - 1)
        return out

    return postprocess(cfg, users, logs, servers, prices, q_true0, q_plan, e_used, p_ref,
                       shap_cache, snapshot, R, T, H, env.now)


def jain(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0 or (x ** 2).sum() == 0:
        return 1.0
    return float(x.sum() ** 2 / (x.size * (x ** 2).sum()))


def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.size == 0 or x.sum() == 0:
        return 0.0
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def postprocess(cfg, users, logs, servers, prices, q_true0, q_plan, e_used, p_ref,
                shap_cache, snapshot, R, T, H, t_end):
    M = len(cfg.models)
    U = len(users)
    eps = cfg.demand_elasticity
    rng = np.random.default_rng([cfg.seed, 77])

    rq = pd.DataFrame(logs.req, columns=["rid", "user", "m", "rnd", "arr", "start", "finish", "bsize", "cold", "om",
                                         "slo"])
    done = set(rq["rid"]) if len(rq) else set()
    pend = [r for s in servers for r in s.queue]
    if pend:
        extra = pd.DataFrame([(r.rid, r.user, r.m, r.rnd, r.arr, t_end, t_end, 0, r.cold, r.om, r.slo) for r in pend
                              if r.rid not in done], columns=rq.columns)
        rq = pd.concat([rq, extra], ignore_index=True)
    rej = pd.DataFrame(logs.rejects, columns=["user", "m", "rnd", "t"])
    rq["lat"] = rq["finish"] - rq["arr"]
    rq["sla"] = rq["slo"].astype(float)
    rq["degraded"] = rq["m"] != rq["om"]
    rq["viol"] = rq["lat"] > rq["sla"]
    rq["price"] = prices[rq["rnd"].to_numpy(), rq["m"].to_numpy()]
    rq["refund"] = cfg.refund_frac * rq["price"] * rq["viol"]

    boots = pd.DataFrame(logs.boots, columns=["m", "t", "cost", "user", "rnd"])
    idle = pd.DataFrame(logs.idle, columns=["m", "t0", "t1", "cost", "user", "rnd"])
    bat = pd.DataFrame(logs.batches, columns=["m", "t", "dur", "n", "cost", "rnd"])
    boot_cost, idle_cost, busy_cost = boots["cost"].sum(), idle["cost"].sum(), bat["cost"].sum()
    alloc_time = (boots["cost"] / boots["m"].map(lambda m: cfg.models[m].gpu_price_h / 3600)).sum() \
        + (idle["t1"] - idle["t0"]).sum() + bat["dur"].sum()

    arr, bcell, icell = group_cell_data(logs, R, M)
    X = np.zeros((R, U))
    Xm = np.zeros(M)
    SA = np.zeros((R, U))
    ir_viol = ir_tot = core_viol = core_tot = 0
    for r in range(R):
        for m in range(M):
            spec = cfg.models[m]
            O = bcell[(r, m)] and sum(b[2] for b in bcell[(r, m)]) or 0.0
            O += sum(d[3] for d in icell[(r, m)])
            tb = arr[(r, m)]
            x = allocate_cell(cfg.allocation, tb, O, bcell[(r, m)], icell[(r, m)], cfg, spec,
                              rng, shap_cache, (r, m))
            for u, v in x.items():
                X[r, u] += v
            Xm[m] += sum(x.values())
            if O <= 0 or not tb:
                continue
            g = CostGame(tb, cfg.keepalive_s, spec.cold_start_s * cfg.cold_scale * spec.gpu_price_h / 3600,
                         spec.gpu_price_h / 3600)
            cN = g.cost(g.players)
            if cN <= 0:
                continue
            k = O / cN
            payers = set(x) | set(g.players)
            tol = 1e-9 * O + 1e-15
            for u in payers:
                sa = k * g.cost([u]) if u in g.iv else 0.0
                SA[r, u] += sa
                ir_tot += 1
                ir_viol += x.get(u, 0.0) > sa + tol
            n = len(g.players)
            if n >= 2 and (cfg.core_samples > 0 or cfg.core_exact_max > 0):
                if n <= cfg.core_exact_max:
                    coals = ([g.players[j] for j in range(n) if mask >> j & 1] for mask in range(1, (1 << n) - 1))
                else:
                    coals = (list(rng.choice(g.players, size=int(rng.integers(1, n)), replace=False))
                             for _ in range(cfg.core_samples))
                for S in coals:
                    core_tot += 1
                    core_viol += sum(x.get(u, 0.0) for u in S) > k * g.cost(S) + tol

    n_ur = np.zeros((R, U))
    np.add.at(n_ur, (rq["rnd"].to_numpy(), rq["user"].to_numpy()), 1)
    late_ur = np.zeros((R, U))
    np.add.at(late_ur, (rq["rnd"].to_numpy(), rq["user"].to_numpy()), rq["viol"].to_numpy(float))
    deg_ur = np.zeros((R, U))
    np.add.at(deg_ur, (rq["rnd"].to_numpy(), rq["user"].to_numpy()), rq["degraded"].to_numpy(float))
    pay_p = np.zeros((R, U))
    np.add.at(pay_p, (rq["rnd"].to_numpy(), rq["user"].to_numpy()), rq["price"].to_numpy())
    ref = np.zeros((R, U))
    np.add.at(ref, (rq["rnd"].to_numpy(), rq["user"].to_numpy()), rq["refund"].to_numpy())
    pay = pay_p + X - ref
    util = np.zeros((R, U))
    for r in range(R):
        for u in users:
            util[r, u.uid] = utility(n_ur[r, u.uid], q_true0[r, u.uid], pay[r, u.uid], p_ref[u.m], eps) \
                - (1 - cfg.fallback_value) * p_ref[u.m] * deg_ur[r, u.uid] \
                - (1 - cfg.late_value) * p_ref[u.m] * late_ur[r, u.uid]

    revenue = rq["price"].sum() - rq["refund"].sum()
    profit = revenue - busy_cost
    total_pay = pay.sum()
    n_req = len(rq)

    per_user = pd.DataFrame({
        "user": [u.uid for u in users], "model": [cfg.models[u.m].name for u in users],
        "base_rate_per_min": [u.q0_rate * 60 for u in users],
        "requests": n_ur.sum(0),
        "rejected": [int((rej["user"] == u.uid).sum()) if len(rej) else 0 for u in users],
        "overhead_paid": X.sum(0), "standalone_cost": SA.sum(0),
        "price_paid": pay_p.sum(0), "refunds": ref.sum(0), "total_paid": pay.sum(0),
        "utility": util.sum(0),
    })
    cvm, iso = [], []
    for u in users:
        ts = np.sort(rq.loc[rq["user"] == u.uid, "arr"].to_numpy())
        g = np.diff(ts)
        cvm.append(float(g.std() / g.mean()) if g.size >= 3 and g.mean() > 0 else np.nan)
        iso.append(float(np.mean(g > cfg.keepalive_s)) if g.size >= 1 else np.nan)
    per_user["cv_measured"] = cvm
    per_user["isolation_ratio"] = iso
    per_user["overhead_per_req"] = per_user["overhead_paid"] / per_user["requests"].replace(0, np.nan)
    per_user["cost_to_standalone"] = per_user["overhead_paid"] / per_user["standalone_cost"].replace(0, np.nan)
    act = per_user[per_user["requests"] > 0]
    ratio = act["cost_to_standalone"].dropna()
    corr = np.nan
    if sps is not None and act["isolation_ratio"].notna().sum() > 3:
        corr = float(sps.spearmanr(act["overhead_per_req"], act["isolation_ratio"], nan_policy="omit")[0])

    pm = []
    for m, s in enumerate(cfg.models):
        d = rq[rq["m"] == m]
        b = bat[bat["m"] == m]
        bt = boots[boots["m"] == m]
        it = idle[idle["m"] == m]
        alloc_m = len(bt) * s.cold_start_s * cfg.cold_scale + (it["t1"] - it["t0"]).sum() + b["dur"].sum()
        nrej = int((rej["m"] == m).sum()) if len(rej) else 0
        pm.append({"model": s.name, "requests": len(d), "rejected": nrej,
                   "degraded_out": int((rq["degraded"] & (rq["om"] == m)).sum()),
                   "sla_miss_incl_rejects_%": 100 * (d["viol"].sum() + nrej) / max(1, len(d) + nrej),
                   "sla_violation_%": 100 * d["viol"].mean() if len(d) else np.nan,
                   "p50_latency_s": d["lat"].quantile(.5) if len(d) else np.nan,
                   "p95_latency_s": d["lat"].quantile(.95) if len(d) else np.nan,
                   "p99_latency_s": d["lat"].quantile(.99) if len(d) else np.nan,
                   "cold_request_%": 100 * d["cold"].mean() if len(d) else np.nan,
                   "boots": len(bt), "avg_batch": b["n"].mean() if len(b) else np.nan,
                   "gpu_util_%": 100 * b["dur"].sum() / alloc_m if alloc_m > 0 else np.nan,
                   "avg_price_per_1k_$": 1000 * d["price"].mean() if len(d) else np.nan,
                   "user_cost_per_1k_$": 1000 * (d["price"].sum() - d["refund"].sum() + Xm[m]) / len(d)
                   if len(d) else np.nan,
                   "overhead_cost_$": bt["cost"].sum() + it["cost"].sum(),
                   "busy_cost_$": b["cost"].sum()})
    per_model = pd.DataFrame(pm)
    q0m = np.array([sum(u.q0_rate for u in users if u.m == m) for m in range(M)])
    wmix = q0m / q0m.sum()
    cpm = per_model["user_cost_per_1k_$"].to_numpy(float)
    ok = np.isfinite(cpm)
    mix_cost = float((wmix[ok] * cpm[ok]).sum() / wmix[ok].sum()) if ok.any() else np.nan

    neg_users = clipped_mass = np.nan
    if cfg.allocation == "shapley" and shap_cache:
        vals = np.concatenate([np.fromiter(v.values(), float) for v in shap_cache.values() if v])
        if vals.size:
            neg_users = 100.0 * float((vals < 0).mean())
            pos = vals[vals > 0].sum()
            clipped_mass = 100.0 * float(-vals[vals < 0].sum() / pos) if pos > 0 else np.nan

    summary = {
        "requests_served": n_req,
        "user_payment_$": total_pay,
        "cost_per_1k_req_$": 1000 * total_pay / max(1, n_req),
        "mix_adj_cost_per_1k_$": mix_cost,
        "avg_price_per_1k_$": 1000 * rq["price"].mean() if n_req else np.nan,
        "provider_revenue_$": revenue,
        "provider_profit_$": profit,
        "gpu_cost_$": boot_cost + idle_cost + busy_cost,
        "boot_cost_$": boot_cost, "idle_cost_$": idle_cost, "busy_cost_$": busy_cost,
        "overhead_share_%": 100 * (boot_cost + idle_cost) / max(1e-12, boot_cost + idle_cost + busy_cost),
        "consumer_utility_$": util.sum(),
        "social_welfare_$": util.sum() + profit,
        "sla_violation_%": 100 * rq["viol"].mean(),
        "rejection_%": 100 * len(rej) / max(1, n_req + len(rej)),
        "degraded_%": 100 * rq["degraded"].sum() / max(1, n_req + len(rej)),
        "sla_miss_incl_rejects_%": 100 * (rq["viol"].sum() + len(rej)) / max(1, n_req + len(rej)),
        "mean_norm_latency": float((rq["lat"] / rq["sla"]).mean()),
        "p95_norm_latency": float((rq["lat"] / rq["sla"]).quantile(.95)),
        "p99_norm_latency": float((rq["lat"] / rq["sla"]).quantile(.99)),
        "cold_request_%": 100 * rq["cold"].mean(),
        "n_boots": len(boots),
        "gpu_util_%": 100 * bat["dur"].sum() / max(1e-9, alloc_time),
        "avg_batch_size": bat["n"].mean() if len(bat) else np.nan,
        "jain_cost_to_standalone": jain(ratio),
        "gini_overhead_per_req": gini(act["overhead_per_req"].fillna(0)),
        "ir_violation_%": 100 * ir_viol / max(1, ir_tot),
        "core_violation_%": 100 * core_viol / max(1, core_tot),
        "spearman_overhead_vs_isolation": corr,
        "active_users": int((per_user["requests"] > 0).sum()),
        "shapley_negative_users_%": neg_users,
        "shapley_clipped_mass_%": clipped_mass,
    }

    out = {"summary": summary, "per_model": per_model}
    if cfg.keep_details:
        lat_s = rq[["m", "lat", "sla", "cold", "bsize"]].copy()
        if len(lat_s) > 40000:
            lat_s = lat_s.sample(40000, random_state=0)
        out.update(per_user=per_user, latency=lat_s,
                   batch_sizes=bat["n"].value_counts().sort_index(),
                   prices=pd.DataFrame(prices, columns=[s.name for s in cfg.models]),
                   demand=pd.DataFrame({"round": np.arange(R), "planned": q_plan.sum(1),
                                        "base": q_true0.sum(1), "realised": n_ur.sum(1)}),
                   snapshot=snapshot,
                   arrivals=rq[["user", "m", "arr"]].copy())
    return out


_P = dict(pricing="stackelberg", allocation="shapley", batching="sla_aware", tariff="two_part",
          admission=True, admission_slack=-0.2, fallback=True, prewarm=False)
MAIN_METHODS = {
    "Industry (Fixed+Trigger)": dict(pricing="fixed", allocation="trigger", batching="fixed", tariff="linear"),
    "Industry + L3": dict(pricing="fixed", allocation="trigger", batching="sla_aware", tariff="linear",
                          admission=True, admission_slack=-0.2, fallback=True),
    "Stackelberg-Prop [prior]": dict(pricing="stackelberg", allocation="proportional", batching="fixed",
                                     tariff="linear"),
    "Stackelberg-Prop + L3": dict(pricing="stackelberg", allocation="proportional", batching="sla_aware",
                                  tariff="linear", admission=True, admission_slack=-0.2, fallback=True),
    "Proposed": dict(_P),
}
ABLATIONS = {
    "Proposed": dict(_P),
    "w/o Shapley -> Equal": {**_P, "allocation": "equal"},
    "w/o Shapley -> Proportional": {**_P, "allocation": "proportional"},
    "w/o Shapley -> Trigger": {**_P, "allocation": "trigger"},
    "w/o two-part -> Linear tariff": {**_P, "tariff": "linear"},
    "w/o capacity control": {**_P, "admission": False, "fallback": False},
    "w/o fallback (admission only)": {**_P, "fallback": False},
    "w/o SLA-batch -> Fixed batch": {**_P, "batching": "fixed"},
    "w/o SLA-batch -> No batch": {**_P, "batching": "none"},
    "w/o Stackelberg -> Fixed price": {**_P, "pricing": "fixed"},
}
SENSITIVITY = {
    "load_scale": [0.25, 0.5, 1.0, 2.0],
    "keepalive_s": [15.0, 60.0, 120.0, 300.0],
    "sla_scale": [0.5, 1.0, 2.0],
    "cold_scale": [0.5, 1.0, 2.0],
    "cs_weight": [0.0, 0.5, 1.0],
    "demand_elasticity": [0.6, 1.2, 2.0],
    "fallback_value": [0.4, 0.7, 1.0],
    "late_value": [0.0, 0.5, 1.0],
}
SENS_LABEL = {"load_scale": "Load scale (x)", "keepalive_s": "Keep-alive window (s)",
              "sla_scale": "SLO scale (x)", "cold_scale": "Cold-start time scale (x)",
              "cs_weight": "Leader weight on consumer surplus",
              "demand_elasticity": "Price elasticity of demand", "fallback_value": "Value of a fallback answer",
              "late_value": "Value of a late answer"}

KEY_METRICS = ["mix_adj_cost_per_1k_$", "provider_profit_$", "social_welfare_$", "sla_miss_incl_rejects_%",
               "p99_norm_latency", "cold_request_%", "gpu_util_%", "avg_batch_size",
               "jain_cost_to_standalone", "ir_violation_%", "core_violation_%",
               "sla_violation_%", "rejection_%", "degraded_%", "cost_per_1k_req_$"]
METRIC_LABEL = {
    "cost_per_1k_req_$": "User cost per 1k req ($)", "provider_profit_$": "Provider profit ($)",
    "social_welfare_$": "Social welfare ($)", "sla_violation_%": "SLA violation (%)",
    "p99_norm_latency": "P99 latency / SLO", "cold_request_%": "Cold-start requests (%)",
    "gpu_util_%": "GPU utilisation (%)", "avg_batch_size": "Avg. batch size",
    "jain_cost_to_standalone": "Jain index (cost / stand-alone)",
    "ir_violation_%": "Individual-rationality violations (%)",
    "core_violation_%": "Core violations (%)", "consumer_utility_$": "Consumer utility ($)",
    "rejection_%": "Rejected by admission (%)",
    "mix_adj_cost_per_1k_$": "Mix-adjusted user cost per 1k ($)",
    "degraded_%": "Served by fallback model (%)",
    "sla_miss_incl_rejects_%": "SLA miss incl. rejected (%)",
}


def _job(args):
    group, label, cfg = args
    t0 = time.time()
    res = simulate(cfg)
    res["group"], res["label"], res["seed"] = group, label, cfg.seed
    res["cfg"] = {k: v for k, v in asdict(cfg).items() if k != "models"}
    res["elapsed_s"] = time.time() - t0
    return res


def build_jobs(base: Config, seeds_main, seeds_sens, which):
    jobs = []
    if which in ("all", "core", "main"):
        for lab, ov in MAIN_METHODS.items():
            for s in seeds_main:
                jobs.append(("main", lab, replace(base, seed=s, keep_details=(s == seeds_main[0]), **ov)))
    if which in ("all", "core", "ablation"):
        for lab, ov in ABLATIONS.items():
            if lab == "Proposed" and which in ("all", "core"):
                continue
            for s in seeds_main:
                jobs.append(("ablation", lab, replace(base, seed=s, keep_details=(s == seeds_main[0]), **ov)))
    if which in ("all", "sensitivity"):
        for par, vals in SENSITIVITY.items():
            for v in vals:
                for lab, ov in MAIN_METHODS.items():
                    for s in seeds_sens:
                        jobs.append((f"sens:{par}={v}", lab, replace(base, seed=s, core_samples=0, core_exact_max=0,
                                                                     **{par: v}, **ov)))
    return jobs


def per_seed_df(results, group_pred):
    rows = []
    for r in results:
        if group_pred(r["group"]):
            d = {"group": r["group"], "method": r["label"], "seed": r["seed"]}
            d.update(r["summary"])
            rows.append(d)
    return pd.DataFrame(rows)


def mean_ci(df, by, metrics):
    g = df.groupby(by, sort=False)
    out = g[metrics].mean()
    n = g.size()
    sd = g[metrics].std(ddof=1).fillna(0)
    tcrit = np.array([sps.t.ppf(0.975, k - 1) if (sps is not None and k > 1) else 1.96 for k in n])
    ci = sd.mul(tcrit, axis=0).div(np.sqrt(n), axis=0)
    res = out.copy()
    for c in metrics:
        res[c + " (±95%CI)"] = ci[c]
    res.insert(0, "n_seeds", n)
    return res.reset_index()


def stat_tests(df, ref, metrics):
    rows = []
    a = df[df["method"] == ref].set_index("seed")
    for meth in df["method"].unique():
        if meth == ref:
            continue
        b = df[df["method"] == meth].set_index("seed")
        common = a.index.intersection(b.index)
        for mt in metrics:
            x, y = a.loc[common, mt].astype(float), b.loc[common, mt].astype(float)
            diff = x - y
            p = np.nan
            if sps is not None and len(common) >= 2 and diff.std() > 0:
                p = float(sps.ttest_rel(x, y).pvalue)
            rows.append({"proposed_vs": meth, "metric": mt, "n": len(common),
                         "mean_proposed": x.mean(), "mean_other": y.mean(),
                         "diff": diff.mean(),
                         "rel_change_%": 100 * diff.mean() / abs(y.mean()) if y.mean() != 0 else np.nan,
                         "paired_t_p_value": p,
                         "significant_0.05": (p < 0.05) if p == p else None})
    return pd.DataFrame(rows)


def workload_tables(cfg: Config, details: dict | None):
    rows = []
    users = make_users(cfg)
    for u in users:
        rows.append({"user": u.uid, "model": cfg.models[u.m].name, "base_rate_per_min": u.q0_rate * 60,
                     "cv_config": u.cv})
    wl = pd.DataFrame(rows)
    stats = {"n_users": cfg.n_users, "workload": cfg.workload,
             "mean_rate_per_min": wl["base_rate_per_min"].mean(),
             "median_rate_per_min": wl["base_rate_per_min"].median(),
             "users_below_1_req_per_min_%": 100 * (wl["base_rate_per_min"] < 1).mean()}
    tr = load_llm_token_works(cfg.azure_llm_path)
    if tr is not None:
        a = tr["arrival_s"]
        g = np.diff(a)
        stats.update({"llm_trace_requests": a.size, "llm_trace_duration_s": a[-1] - a[0],
                      "llm_trace_rate_per_s": a.size / max(1, a[-1] - a[0]),
                      "llm_trace_interarrival_cv": g.std() / g.mean(),
                      "llm_ctx_tokens_mean": float(np.mean(tr["ctx"])),
                      "llm_gen_tokens_mean": float(np.mean(tr["gen"])),
                      "llm_work_cv": float(tr["work"].std())})
    return wl, pd.DataFrame([stats]).T.reset_index().rename(columns={"index": "statistic", 0: "value"})


COL = {"Industry (Fixed+Trigger)": "#8c8c8c", "Stackelberg-Prop [prior]": "#4c72b0", "Proposed": "#c44e52",
       "Industry + L3": "#3a3a3a", "Stackelberg-Prop + L3": "#9ecae1",
       "w/o Shapley -> Equal": "#dd8452", "w/o Shapley -> Proportional": "#937860",
       "w/o Shapley -> Trigger": "#64b5cd", "w/o SLA-batch -> Fixed batch": "#55a868",
       "w/o SLA-batch -> No batch": "#8172b3", "w/o Stackelberg -> Fixed price": "#da8bc3",
       "w/o two-part -> Linear tariff": "#ccb974", "w/o capacity control": "#4d4d4d",
       "w/o fallback (admission only)": "#a6a6a6"}
ALLOC_NAME = {"Proposed": "Shapley", "w/o Shapley -> Equal": "Equal",
              "w/o Shapley -> Proportional": "Proportional", "w/o Shapley -> Trigger": "Trigger-pays"}


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 110,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "savefig.bbox": "tight"})


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=300)
    fig.savefig(os.path.join(outdir, name + ".pdf"))
    plt.close(fig)


def _short(lbl):
    return lbl.replace(" [prior]", "\n[prior]").replace(" (Fixed+Trigger)", "\n(Fixed+Trigger)") \
        .replace(" -> ", "\n-> ")


def fig_bars(summ, methods, metrics, outdir, name, ncols=4):
    n = len(metrics)
    nr = int(math.ceil(n / ncols))
    fig, axs = plt.subplots(nr, ncols, figsize=(3.0 * ncols, 2.6 * nr))
    axs = np.atleast_1d(axs).ravel()
    s = summ.set_index("method").loc[methods]
    for ax, mt in zip(axs, metrics):
        vals, err = s[mt].to_numpy(float), s[mt + " (±95%CI)"].to_numpy(float)
        ax.bar(range(len(methods)), vals, yerr=err, capsize=3, color=[COL.get(m, "#999") for m in methods],
               edgecolor="black", linewidth=0.4)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([_short(m) for m in methods], fontsize=6.5, rotation=0)
        ax.set_title(METRIC_LABEL.get(mt, mt))
    for ax in axs[n:]:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, outdir, name)


def fig_latency_cdf(details, methods, outdir):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for m in methods:
        d = details[m]["latency"]
        x = np.sort((d["lat"] / d["sla"]).to_numpy())
        y = np.arange(1, x.size + 1) / x.size
        axs[0].plot(x, y, label=m, color=COL.get(m), lw=1.4)
        axs[1].plot(x, 1 - y + 1e-6, label=m, color=COL.get(m), lw=1.4)
    for ax in axs:
        ax.axvline(1.0, ls="--", color="k", lw=0.8)
        ax.set_xlabel("Latency / SLO")
        ax.set_xscale("log")
    axs[0].set_ylabel("CDF")
    axs[1].set_ylabel("CCDF (tail)")
    axs[1].set_yscale("log")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Normalised end-to-end latency")
    axs[1].set_title("Tail latency")
    fig.tight_layout()
    _save(fig, outdir, "fig03_latency_cdf")


def fig_ablation_heatmap(abl_summ, outdir):
    ms = ["mix_adj_cost_per_1k_$", "provider_profit_$", "social_welfare_$", "sla_miss_incl_rejects_%",
          "p99_norm_latency", "gpu_util_%", "jain_cost_to_standalone", "ir_violation_%", "core_violation_%"]
    higher_better = {"provider_profit_$", "social_welfare_$", "gpu_util_%", "jain_cost_to_standalone"}
    pp_cols = {"ir_violation_%", "core_violation_%"}
    s = abl_summ.set_index("method")
    ref = s.loc["Proposed", ms].astype(float)
    rows = [m for m in s.index if m != "Proposed"]
    mat = np.full((len(rows), len(ms)), np.nan)
    txt = [[""] * len(ms) for _ in rows]
    for i, r in enumerate(rows):
        for j, c in enumerate(ms):
            v = float(s.loc[r, c])
            if c in pp_cols:
                d = v - ref[c]
                mat[i, j] = 10 * d
                txt[i][j] = f"{d:+.2f} pp"
            elif abs(ref[c]) > 1e-12:
                d = 100 * (v - ref[c]) / abs(ref[c])
                mat[i, j] = d
                txt[i][j] = f"{d:+.0f}%"
    good = mat * np.array([1 if c in higher_better else -1 for c in ms])[None, :]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.8))
    lim = np.nanpercentile(np.abs(good), 90) if np.isfinite(good).any() else 1
    lim = max(lim, 1e-6)
    im = ax.imshow(np.clip(good, -lim, lim), cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([METRIC_LABEL[m] for m in ms], rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(ms)):
            ax.text(j, i, txt[i][j] or "n/a", ha="center", va="center", fontsize=7)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, ticks=[-lim, 0, lim])
    cb.ax.set_yticklabels(["worse", "same", "better"])
    ax.set_xlabel("cell text = change w.r.t. Proposed (red = variant is worse than Proposed)")
    ax.set_title("Ablation study: relative change when one component is replaced")
    fig.tight_layout()
    _save(fig, outdir, "fig04_ablation_heatmap")


def fig_stackelberg(details, cfg, outdir):
    d = details["Proposed"]
    sn = d["snapshot"]
    M = len(cfg.models)
    eps = cfg.demand_elasticity
    fig, axs = plt.subplots(1, M + 1, figsize=(3.2 * (M + 1), 3.0))
    for m in range(M):
        ids = np.array(sn["by_model"][m])
        pr = sn["p_ref"][m]
        grid = np.linspace(sn["c_hat"][m], pr * (1 + 1 / eps) * 1.2, 300)
        prof = leader_objective(grid, sn["q0_hat"][ids], sn["e_hat"][ids], pr, eps, sn["c_hat"][m],
                                sn["v_hat"][m], cfg.refund_frac, cfg.cs_weight, sn["f_hat"][ids])
        dem = follower_response(sn["q0_hat"][ids][None, :], grid[:, None], sn["e_hat"][ids][None, :],
                                sn["f_hat"][ids][None, :], pr, eps)[0].sum(1)
        ax = axs[m]
        ax.plot(grid * 1000, prof, color="#c44e52", label="Leader objective")
        ax.axvline(sn["p"][m] * 1000, ls="--", color="k", lw=0.8, label="p* (equilibrium)")
        ax.axvline(sn["p_fix"][m] * 1000, ls=":", color="#8c8c8c", lw=0.9, label="fixed (cost-plus) price")
        ax2 = ax.twinx()
        ax2.plot(grid * 1000, dem, color="#4c72b0", lw=1.0, label="Followers' demand")
        ax2.set_ylabel("Total demand (req/round)", color="#4c72b0", fontsize=8)
        ax2.grid(False)
        ax.set_xlabel("Price ($ per 1k requests)")
        ax.set_ylabel("Leader objective ($/round)")
        ax.set_title(cfg.models[m].name)
        if m == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="lower center")
    ax = axs[M]
    pr = d["prices"]
    for c in pr.columns:
        ax.plot(np.arange(len(pr)), pr[c] / pr[c].iloc[0], label=c)
    ax.set_xlabel("Pricing round")
    ax.set_ylabel("p* / p*(round 0)")
    ax.set_title("Equilibrium price over rounds")
    ax.legend()
    fig.tight_layout()
    _save(fig, outdir, "fig05_stackelberg_equilibrium")


def fig_allocation(details, outdir):
    labs = [l for l in ALLOC_NAME if l in details]
    fig, axs = plt.subplots(1, len(labs) + 1, figsize=(3.0 * (len(labs) + 1), 3.0), sharey=False)
    for ax, l in zip(axs, labs):
        pu = details[l]["per_user"]
        pu = pu[pu["requests"] > 0]
        sc = ax.scatter(pu["isolation_ratio"], 1000 * pu["overhead_per_req"], c=np.log10(pu["base_rate_per_min"]),
                        cmap="viridis", s=18, edgecolor="k", linewidth=0.3)
        ax.set_yscale("log")
        ax.set_xlabel("Isolation ratio (gaps > keep-alive)")
        ax.set_title(ALLOC_NAME[l])
    axs[0].set_ylabel("Cold/idle overhead paid ($ per 1k req)")
    fig.colorbar(sc, ax=axs[len(labs) - 1], label="log10 rate (req/min)")
    ax = axs[-1]
    data = []
    for l in labs:
        pu = details[l]["per_user"]
        data.append(pu["cost_to_standalone"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy())
    ax.boxplot(data, showfliers=True, flierprops=dict(markersize=2))
    ax.set_xticks(range(1, len(labs) + 1))
    ax.set_xticklabels([ALLOC_NAME[l] for l in labs], rotation=20)
    ax.axhline(1.0, color="r", ls="--", lw=0.8, label="stand-alone cost (IR bound)")
    ax.set_yscale("log")
    ax.set_ylabel("Overhead paid / stand-alone cost")
    ax.set_title("Individual rationality")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    _save(fig, outdir, "fig06_cost_allocation_per_user")


def fig_fairness(abl_summ, outdir):
    labs = [l for l in ALLOC_NAME if l in set(abl_summ["method"])]
    s = abl_summ.set_index("method").loc[labs]
    mts = ["jain_cost_to_standalone", "ir_violation_%", "core_violation_%", "gini_overhead_per_req"]
    names = {**METRIC_LABEL, "gini_overhead_per_req": "Gini of overhead per request"}
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.8))
    for ax, mt in zip(axs, mts):
        err = s[mt + " (±95%CI)"] if mt + " (±95%CI)" in s else None
        ax.bar(range(len(labs)), s[mt], yerr=err, capsize=3, color=[COL[l] for l in labs],
               edgecolor="k", linewidth=0.4)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels([ALLOC_NAME[l] for l in labs], rotation=20)
        ax.set_title(names[mt])
    fig.tight_layout()
    _save(fig, outdir, "fig07_fairness_mechanisms")


def fig_sensitivity(sens_summ, outdir):
    pars = list(SENSITIVITY)
    mts = ["mix_adj_cost_per_1k_$", "sla_miss_incl_rejects_%", "social_welfare_$"]
    fig, axs = plt.subplots(len(mts), len(pars), figsize=(3.1 * len(pars), 2.5 * len(mts)))
    for j, par in enumerate(pars):
        sub = sens_summ[sens_summ["param"] == par]
        for i, mt in enumerate(mts):
            ax = axs[i, j]
            for meth in MAIN_METHODS:
                d = sub[sub["method"] == meth].sort_values("value")
                ax.errorbar(d["value"], d[mt], yerr=d[mt + " (±95%CI)"], marker="o", ms=3, capsize=2,
                            label=meth, color=COL[meth], lw=1.2)
            if par in ("load_scale", "keepalive_s"):
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(SENSITIVITY[par]))
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            if i == len(mts) - 1:
                ax.set_xlabel(SENS_LABEL[par])
            if j == 0:
                ax.set_ylabel(METRIC_LABEL[mt])
    axs[0, 0].legend(fontsize=6.5)
    fig.tight_layout()
    _save(fig, outdir, "fig08_sensitivity")


def fig_pareto(main_summ, abl_summ, outdir):
    s = pd.concat([main_summ, abl_summ]).drop_duplicates("method").set_index("method")
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for meth, row in s.iterrows():
        ax.errorbar(row["sla_miss_incl_rejects_%"], row["mix_adj_cost_per_1k_$"],
                    xerr=row["sla_miss_incl_rejects_% (±95%CI)"], yerr=row["mix_adj_cost_per_1k_$ (±95%CI)"],
                    fmt="o", color=COL.get(meth, "#999"), capsize=2, label=meth, ms=6, mec="k", mew=0.4)
    ax.set_xlabel("SLA miss incl. rejected requests (%)")
    ax.set_ylabel("Mix-adjusted user cost per 1k requests ($)")
    ax.set_title("Cost vs. QoS trade-off (lower-left is better)")
    ax.legend(fontsize=6.3, loc="best")
    fig.tight_layout()
    _save(fig, outdir, "fig09_cost_qos_pareto")


def fig_batching(details, sens_summ, outdir):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.0))
    for l, nm in [("w/o SLA-batch -> No batch", "No batching"), ("w/o SLA-batch -> Fixed batch", "Fixed batch"),
                  ("Proposed", "SLA-aware (proposed)")]:
        if l not in details:
            continue
        bs = details[l]["batch_sizes"]
        reqs = bs.index.to_numpy() * bs.to_numpy()
        axs[0].plot(bs.index, reqs / reqs.sum(), marker="o", ms=3, label=nm, color=COL[l])
    axs[0].set_xlabel("Batch size")
    axs[0].set_ylabel("Fraction of requests")
    axs[0].set_title("Batch-size distribution")
    axs[0].legend()
    if sens_summ is not None and len(sens_summ):
        sub = sens_summ[sens_summ["param"] == "load_scale"]
        for meth in MAIN_METHODS:
            d = sub[sub["method"] == meth].sort_values("value")
            axs[1].errorbar(d["value"], d["gpu_util_%"], yerr=d["gpu_util_% (±95%CI)"], marker="o", ms=3,
                            capsize=2, label=meth, color=COL[meth])
        axs[1].set_xscale("log")
        axs[1].xaxis.set_major_locator(matplotlib.ticker.FixedLocator(SENSITIVITY["load_scale"]))
        axs[1].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axs[1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        axs[1].set_xlabel("Load scale (x)")
        axs[1].set_ylabel("GPU utilisation (%)")
        axs[1].set_title("Utilisation vs. load")
        axs[1].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, outdir, "fig10_batching_utilisation")


def fig_workload(cfg, details, outdir):
    d = details.get("Proposed") or next(iter(details.values()))
    a = d["arrivals"]
    users = make_users(cfg)
    fig, axs = plt.subplots(1, 4, figsize=(13, 2.8))
    for m, s in enumerate(cfg.models):
        t = a.loc[a["m"] == m, "arr"].to_numpy()
        h, e = np.histogram(t, bins=np.arange(0, t.max() + 60, 60)) if t.size else (np.array([]), np.array([0]))
        axs[0].plot(e[:-1] / 3600, h, lw=0.7, label=s.name)
    axs[0].set_xlabel("Time (h)")
    axs[0].set_ylabel("Requests / min")
    axs[0].set_title("Arrival process")
    axs[0].legend(fontsize=6.5)
    rates = np.array([u.q0_rate * 60 for u in users])
    axs[1].hist(rates, bins=np.logspace(np.log10(rates.min()), np.log10(rates.max()), 15), edgecolor="k", lw=0.3)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("User base rate (req/min)")
    axs[1].set_ylabel("Users")
    axs[1].set_title("Demand heterogeneity")
    cv = d["per_user"]["cv_measured"].dropna()
    axs[2].hist(cv, bins=15, edgecolor="k", lw=0.3, color="#55a868")
    axs[2].set_xlabel("Measured inter-arrival CV")
    axs[2].set_title("Burstiness of users")
    tr = load_llm_token_works(cfg.azure_llm_path)
    if tr is not None:
        axs[3].hist(tr["work"], bins=50, color="#8172b3", edgecolor="k", lw=0.2)
        axs[3].set_title("Request work (Azure LLM trace)")
    else:
        rng = np.random.default_rng(0)
        axs[3].hist(sample_works(cfg, rng, cfg.models[1], 20000), bins=50, color="#8172b3", edgecolor="k", lw=0.2)
        axs[3].set_title("Request work (synthetic lognormal)")
    axs[3].set_xlabel("Normalised work per request")
    fig.tight_layout()
    _save(fig, outdir, "fig01_workload")


def fig_cost_breakdown(main_summ, abl_summ, outdir):
    s = pd.concat([main_summ, abl_summ]).drop_duplicates("method").set_index("method")
    meths = list(s.index)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bottom = np.zeros(len(meths))
    for c, colr, nm in [("busy_cost_$", "#4c72b0", "Execution (busy)"), ("boot_cost_$", "#c44e52", "Cold start"),
                        ("idle_cost_$", "#dd8452", "Keep-alive idle")]:
        v = s[c].to_numpy(float)
        ax.bar(range(len(meths)), v, bottom=bottom, color=colr, label=nm, edgecolor="k", lw=0.3)
        bottom += v
    ax.set_xticks(range(len(meths)))
    ax.set_xticklabels([_short(m) for m in meths], fontsize=6.5)
    ax.set_ylabel("Provider GPU cost ($)")
    ax.set_title("GPU cost breakdown")
    ax.legend()
    fig.tight_layout()
    _save(fig, outdir, "fig11_gpu_cost_breakdown")


def shapley_benchmark(seed=0, perms_list=(16, 32, 64, 128), n_exact=(4, 6, 8, 10, 12),
                      n_scale=(10, 20, 40, 80, 160, 320)):
    rng = np.random.default_rng(seed)
    price = 2.5 / 3600
    boot = 12.0 * price

    def random_game(n):
        tb = {}
        for u in range(n):
            q = float(rng.lognormal(math.log(20), 1.0))
            cv = float(np.clip(rng.lognormal(math.log(1.5), 0.6), 0.3, 6))
            tb[u] = gamma_arrivals(rng, q, cv, 900.0)
            if tb[u].size == 0:
                tb[u] = np.array([rng.random() * 900.0])
        return CostGame(tb, 60.0, boot, price)

    acc = []
    for n in n_exact:
        for rep in range(30):
            g = random_game(n)
            t0 = time.perf_counter()
            ex = g.shapley(rng, 2, exact_max=99)
            t_ex = time.perf_counter() - t0
            ev = np.array([ex[u] for u in g.players])
            t0 = time.perf_counter()
            cf = g.shapley_closed_form()
            t_cf = time.perf_counter() - t0
            cv_ = np.array([cf[u] for u in g.players])
            acc.append({"players": n, "rep": rep, "perms": "closed-form",
                        "rel_L1_error_%": 100 * np.abs(cv_ - ev).sum() / max(1e-12, np.abs(ev).sum()),
                        "max_rel_error_%": 0.0, "exact_time_s": t_ex, "mc_time_s": t_cf})
            for P in perms_list:
                t0 = time.perf_counter()
                mc = g.shapley(rng, P, exact_max=0)
                t_mc = time.perf_counter() - t0
                mv = np.array([mc[u] for u in g.players])
                acc.append({"players": n, "rep": rep, "perms": P,
                            "rel_L1_error_%": 100 * np.abs(mv - ev).sum() / max(1e-12, np.abs(ev).sum()),
                            "max_rel_error_%": 100 * float(np.max(np.abs(mv - ev) / np.maximum(np.abs(ev), 1e-12))),
                            "exact_time_s": t_ex, "mc_time_s": t_mc})
    sc = []
    for n in n_scale:
        g = random_game(n)
        t0 = time.perf_counter()
        g.shapley(rng, 64, exact_max=0)
        t_mc = time.perf_counter() - t0
        t0 = time.perf_counter()
        g.shapley_closed_form()
        sc.append({"players": n, "mc64_time_s": t_mc, "closed_form_time_s": time.perf_counter() - t0})
    return pd.DataFrame(acc), pd.DataFrame(sc)


def fig_shapley_benchmark(acc, sc, outdir):
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.0))
    mc = acc[acc["perms"] != "closed-form"].copy()
    mc["perms"] = mc["perms"].astype(int)
    a = mc.groupby(["players", "perms"])["rel_L1_error_%"].mean().reset_index()
    for P, d in a.groupby("perms"):
        axs[0].plot(d["players"], d["rel_L1_error_%"], marker="o", ms=3, label=f"MC, {P} permutations")
    cfe = acc[acc["perms"] == "closed-form"].groupby("players")["rel_L1_error_%"].mean()
    axs[0].plot(cfe.index, cfe.values, marker="*", ms=6, color="k", label="closed form (exact)")
    axs[0].set_xlabel("Players (users of one model)")
    axs[0].set_ylabel("Relative L1 error vs exact (%)")
    axs[0].set_title("Shapley accuracy (vs. exact enumeration)")
    axs[0].legend(fontsize=7)
    t = acc.groupby("players")[["exact_time_s"]].mean().reset_index()
    tm = acc[acc["perms"] == 64].groupby("players")["mc_time_s"].mean().reset_index()
    tc = acc[acc["perms"] == "closed-form"].groupby("players")["mc_time_s"].mean().reset_index()
    axs[1].plot(t["players"], t["exact_time_s"], marker="o", ms=3, label="enumeration (2^n)")
    axs[1].plot(tm["players"], tm["mc_time_s"], marker="s", ms=3, label="Monte-Carlo, 64 perms")
    axs[1].plot(tc["players"], tc["mc_time_s"], marker="*", ms=6, color="k", label="closed form")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Players")
    axs[1].set_ylabel("Time per billing round (s)")
    axs[1].set_title("Exact vs approximate cost")
    axs[1].legend(fontsize=7)
    axs[2].plot(sc["players"], sc["mc64_time_s"], marker="s", ms=3, label="Monte-Carlo, 64 perms")
    axs[2].plot(sc["players"], sc["closed_form_time_s"], marker="*", ms=6, color="k", label="closed form")
    axs[2].legend(fontsize=7)
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("Players")
    axs[2].set_ylabel("Time per billing round (s)")
    axs[2].set_title("Scalability")
    fig.tight_layout()
    _save(fig, outdir, "fig12_shapley_benchmark")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    US = USER_SETTINGS

    def _path(p):
        return p if (not p or os.path.isabs(p)) else os.path.join(here, p)

    workers_default = US["WORKERS"] or (1 if _in_spyder() else max(1, os.cpu_count() or 2))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", default=US["QUICK"], help="short smoke test")
    ap.add_argument("--horizon", type=float, default=US["HORIZON_HOURS"], help="simulated hours per run")
    ap.add_argument("--seeds", type=int, default=US["SEEDS"], help="seeds for main + ablation")
    ap.add_argument("--sens-seeds", type=int, default=US["SENS_SEEDS"], help="seeds for sensitivity sweeps")
    ap.add_argument("--users", type=int, default=US["USERS"])
    ap.add_argument("--workers", type=int, default=workers_default)
    ap.add_argument("--only", choices=["all", "core", "main", "ablation", "sensitivity"], default=US["ONLY"])
    ap.add_argument("--azure-llm", default=_path(US["AZURE_LLM_FILE"]))
    ap.add_argument("--azure-functions", default=_path(US["AZURE_FUNCTIONS_FILE"])
                    if US["WORKLOAD"] == "azure_functions" else "")
    ap.add_argument("--synthetic", action="store_true", help="ignore the Azure Functions trace")
    ap.add_argument("--out", default=_path(US["OUT_DIR"]))
    ap.add_argument("--replot", action="store_true", default=US["REPLOT"],
                    help="re-draw figures / Excel from results/raw_results.pkl without simulating")
    a, _unknown = ap.parse_known_args()
    if a.synthetic:
        a.azure_functions = ""
    if a.azure_llm and not os.path.exists(a.azure_llm):
        print(f"[warning] Azure LLM file not found ({a.azure_llm}) -> synthetic request sizes")
        a.azure_llm = ""
    if a.azure_functions and not os.path.exists(a.azure_functions):
        print("\n[ERROR] Azure Functions trace not found:\n   " + a.azure_functions +
              "\n   Run 'python scripts/download_data.py' (see README.md),\n"
              "   or use --synthetic.\n")
        return

    base = Config()
    if a.horizon:
        base = replace(base, horizon_h=a.horizon)
    if a.quick:
        base = replace(base, horizon_h=1.5, shapley_perms=32, core_samples=60)
        a.seeds, a.sens_seeds = min(a.seeds, 2), 1
    if a.users:
        base = replace(base, n_users=a.users)
    if not a.azure_llm:
        for cand in ("data/AzureLLMInferenceTrace_conv.csv", "AzureLLMInferenceTrace_conv.csv"):
            p = os.path.join(os.path.dirname(os.path.abspath(__file__)), cand)
            if os.path.exists(p):
                a.azure_llm = p
                break
    if a.azure_llm:
        base = replace(base, azure_llm_path=a.azure_llm)
    if a.azure_functions:
        base = replace(base, workload="azure_functions", azure_functions_path=a.azure_functions)

    seeds_main = list(range(1, a.seeds + 1))
    seeds_sens = list(range(1, a.sens_seeds + 1))
    figdir = os.path.join(a.out, "figures")
    os.makedirs(figdir, exist_ok=True)
    raw = os.path.join(a.out, "raw_results.pkl")
    t0 = time.time()
    if a.replot:
        with open(raw, "rb") as f:
            base, results = pickle.load(f)
    else:
        jobs = build_jobs(base, seeds_main, seeds_sens, a.only)
        print(f"[info] {len(jobs)} simulation runs | horizon={base.horizon_h} h | users={base.n_users} | "
              f"workers={a.workers} | LLM trace={'yes' if base.azure_llm_path else 'no (synthetic)'} | "
              f"workload={base.workload}")
        results = []
        pool = Pool(a.workers) if a.workers > 1 else None
        it = pool.imap_unordered(_job, jobs) if pool else map(_job, jobs)
        try:
            for k, r in enumerate(it, 1):
                results.append(r)
                if k % max(1, len(jobs) // 20) == 0 or k == len(jobs):
                    print(f"  {k}/{len(jobs)} runs done  ({time.time() - t0:.0f}s)", flush=True)
        finally:
            if pool:
                pool.close()
                pool.join()
        with open(raw, "wb") as f:
            pickle.dump((base, results), f)

    _style()
    allm = list(results[0]["summary"].keys())
    main_seed = per_seed_df(results, lambda g: g == "main")
    abl_seed = per_seed_df(results, lambda g: g == "ablation")
    if len(main_seed) and a.only in ("all", "core"):
        abl_seed = pd.concat([main_seed[main_seed["method"] == "Proposed"].assign(group="ablation"), abl_seed])
    sens_seed = per_seed_df(results, lambda g: g.startswith("sens:"))
    if len(sens_seed):
        sens_seed["param"] = sens_seed["group"].str.extract(r"sens:(\w+)=")[0]
        sens_seed["value"] = sens_seed["group"].str.extract(r"=([\d.]+)")[0].astype(float)

    order_main = [m for m in MAIN_METHODS if m in set(main_seed.get("method", []))]
    order_abl = [m for m in ABLATIONS if m in set(abl_seed.get("method", []))]
    main_summ = mean_ci(main_seed, ["method"], allm) if len(main_seed) else pd.DataFrame()
    abl_summ = mean_ci(abl_seed, ["method"], allm) if len(abl_seed) else pd.DataFrame()
    sens_summ = mean_ci(sens_seed, ["param", "value", "method"], allm) if len(sens_seed) else pd.DataFrame()

    details = {}
    for r in results:
        if "per_user" in r and r["label"] not in details:
            details[r["label"]] = r

    cfg_fig = details["Proposed"]["cfg"] if "Proposed" in details else None
    if details:
        fig_workload(base, details, figdir)
    if len(main_summ):
        fig_bars(main_summ, order_main, KEY_METRICS[:8], figdir, "fig02_main_comparison")
        fig_bars(main_summ, order_main, ["jain_cost_to_standalone", "ir_violation_%", "core_violation_%",
                                         "consumer_utility_$", "rejection_%", "degraded_%"],
                 figdir, "fig02b_main_fairness_welfare", ncols=3)
        fig_latency_cdf(details, [m for m in order_main if m in details], figdir)
    if len(abl_summ) and "Proposed" in set(abl_summ["method"]):
        fig_ablation_heatmap(abl_summ, figdir)
        fig_bars(abl_summ, order_abl, KEY_METRICS[:8], figdir, "fig04b_ablation_bars")
        fig_fairness(abl_summ, figdir)
    if "Proposed" in details:
        fig_stackelberg(details, base, figdir)
    if all(l in details for l in ALLOC_NAME):
        fig_allocation(details, figdir)
    if len(sens_summ):
        fig_sensitivity(sens_summ, figdir)
    if len(main_summ) and len(abl_summ):
        fig_pareto(main_summ, abl_summ, figdir)
        fig_cost_breakdown(main_summ, abl_summ, figdir)
    if details:
        fig_batching(details, sens_summ if len(sens_summ) else None, figdir)
    sb_acc, sb_sc = shapley_benchmark()
    fig_shapley_benchmark(sb_acc, sb_sc, figdir)

    xlsx = os.path.join(a.out, "results.xlsx")
    wl, wl_stats = workload_tables(base, details)
    cfg_rows = [{"parameter": k, "value": str(v)} for k, v in asdict(base).items() if k != "models"]
    model_rows = pd.DataFrame([asdict(s) for s in base.models])
    readme = pd.DataFrame({"sheet": [
        "Config", "Models", "Main_Summary", "Main_PerSeed", "Main_StatTests", "Main_PerModel",
        "Ablation_Summary", "Ablation_PerSeed", "Ablation_DeltaPct", "Ablation_StatTests",
        "Sensitivity_Summary", "Sensitivity_PerSeed", "PerUser_<method>", "Prices_Proposed",
        "Demand_Proposed", "Latency_Percentiles", "Workload_Users", "Workload_Stats", "Metric_Definitions"],
        "content": [
        "All simulation parameters", "AI model / GPU specifications (cost, cold start, latency model, SLO)",
        "Main comparison: mean and 95% CI over seeds", "Main comparison: every seed",
        "Paired t-tests: Proposed vs. each baseline", "Per-model metrics (first seed)",
        "Ablation: mean and 95% CI", "Ablation: every seed", "Ablation: % change w.r.t. Proposed",
        "Paired t-tests: Proposed vs. each ablation variant",
        "Sensitivity sweeps (load, keep-alive, SLO, cold start): mean and 95% CI",
        "Sensitivity sweeps: every seed", "Per-user cost allocation and utility (first seed)",
        "Equilibrium price per round and model (first seed)", "Planned / realised demand per round",
        "Latency percentiles per method and model (first seed)", "Generated user population",
        "Workload statistics (and Azure LLM trace statistics if used)", "Definitions of all metrics"]})
    defs = pd.DataFrame([
        ("cost_per_1k_req_$", "Total user payment (price + allocated overhead - SLA refunds) per 1000 requests"),
        ("provider_profit_$", "Revenue (price - refunds) minus GPU execution cost; cold/idle cost is passed to users"),
        ("social_welfare_$", "Sum of users' quasi-linear utilities + provider profit"),
        ("sla_violation_%", "Share of requests whose end-to-end latency exceeds the model SLO"),
        ("p99_norm_latency", "99th percentile of latency / SLO"),
        ("cold_request_%", "Share of requests that waited for a cold start"),
        ("gpu_util_%", "Busy GPU time / allocated GPU time (boot + busy + idle)"),
        ("jain_cost_to_standalone", "Jain index of (overhead paid / stand-alone cost) across users: "
                                     "1 = every user gets the same relative saving from sharing"),
        ("ir_violation_%", "User-rounds where a user pays more than its stand-alone cost (individual rationality)"),
        ("core_violation_%", "Sampled coalitions that pay more than their own stand-alone cost (core stability)"),
        ("spearman_overhead_vs_isolation", "Rank correlation between overhead paid per request and the user's "
                                            "isolation ratio (share of gaps > keep-alive): cost-causation fidelity"),
        ("gini_overhead_per_req", "Gini coefficient of overhead paid per request across active users"),
    ], columns=["metric", "definition"])

    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        pd.DataFrame(cfg_rows).to_excel(xw, sheet_name="Config", index=False)
        model_rows.to_excel(xw, sheet_name="Models", index=False)
        if len(main_summ):
            main_summ.to_excel(xw, sheet_name="Main_Summary", index=False)
            main_seed.to_excel(xw, sheet_name="Main_PerSeed", index=False)
            stat_tests(main_seed, "Proposed", KEY_METRICS).to_excel(xw, sheet_name="Main_StatTests", index=False)
            pm = []
            for meth in order_main:
                if meth in details:
                    pm.append(details[meth]["per_model"].assign(method=meth))
            if pm:
                pd.concat(pm).to_excel(xw, sheet_name="Main_PerModel", index=False)
        if len(abl_summ):
            abl_summ.to_excel(xw, sheet_name="Ablation_Summary", index=False)
            abl_seed.to_excel(xw, sheet_name="Ablation_PerSeed", index=False)
            s = abl_summ.set_index("method")
            if "Proposed" in s.index:
                ref = s.loc["Proposed", allm].astype(float)
                dl = (s[allm].astype(float) - ref) / ref.abs().replace(0, np.nan) * 100
                dl.reset_index().to_excel(xw, sheet_name="Ablation_DeltaPct", index=False)
                stat_tests(abl_seed, "Proposed", KEY_METRICS).to_excel(xw, sheet_name="Ablation_StatTests", index=False)
        if len(sens_summ):
            sens_summ.to_excel(xw, sheet_name="Sensitivity_Summary", index=False)
            sens_seed.to_excel(xw, sheet_name="Sensitivity_PerSeed", index=False)
        for meth, d in details.items():
            nm = ("PerUser_" + meth.replace("w/o ", "wo_").replace(" -> ", "_").replace(" ", "")
                  .replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("+", ""))[:31]
            d["per_user"].to_excel(xw, sheet_name=nm, index=False)
        if "Proposed" in details:
            details["Proposed"]["prices"].mul(1000).rename(columns=lambda c: c + " ($/1k req)") \
                .rename_axis("round").reset_index().to_excel(xw, sheet_name="Prices_Proposed", index=False)
            details["Proposed"]["demand"].to_excel(xw, sheet_name="Demand_Proposed", index=False)
        lp = []
        for meth, d in details.items():
            L = d["latency"]
            for m, s_ in enumerate(base.models):
                x = L.loc[L["m"] == m, "lat"]
                if len(x):
                    lp.append({"method": meth, "model": s_.name, "nominal_SLO_s": max(s_.slo_factor * s_.t_iso_s, s_.slo_floor_s) * base.sla_scale,
                               **{f"p{q}": x.quantile(q / 100) for q in (50, 90, 95, 99, 99.9)},
                               "mean": x.mean(), "max": x.max()})
        if lp:
            pd.DataFrame(lp).to_excel(xw, sheet_name="Latency_Percentiles", index=False)
        wl.to_excel(xw, sheet_name="Workload_Users", index=False)
        wl_stats.to_excel(xw, sheet_name="Workload_Stats", index=False)
        defs.to_excel(xw, sheet_name="Metric_Definitions", index=False)
        sb_acc.to_excel(xw, sheet_name="Shapley_Accuracy", index=False)
        sb_sc.to_excel(xw, sheet_name="Shapley_Scalability", index=False)

    pd.set_option("display.width", 200)
    cols = ["method", "cost_per_1k_req_$", "mix_adj_cost_per_1k_$", "provider_profit_$", "social_welfare_$",
            "sla_violation_%", "rejection_%", "degraded_%", "sla_miss_incl_rejects_%", "gpu_util_%", "avg_batch_size", "jain_cost_to_standalone", "ir_violation_%", "core_violation_%"]
    if len(main_summ):
        print("\n=== Main comparison (mean over seeds) ===")
        print(main_summ[cols].round(4).to_string(index=False))
    if len(abl_summ):
        print("\n=== Ablation (mean over seeds) ===")
        print(abl_summ[cols].round(4).to_string(index=False))
    print(f"\n[done] {time.time() - t0:.0f}s  ->  {xlsx}\n        figures -> {figdir}")
    with open(os.path.join(a.out, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"finished: {time.strftime('%Y-%m-%d %H:%M')}   elapsed: {time.time() - t0:.0f} s\n")
        fh.write(f"settings: only={a.only} seeds={a.seeds} sens_seeds={a.sens_seeds} horizon={base.horizon_h}h "
                 f"users={base.n_users} quick={a.quick} workers={a.workers}\n")
        fh.write(f"workload={base.workload}  functions_file={os.path.basename(base.azure_functions_path) or '-'}  "
                 f"llm_file={os.path.basename(base.azure_llm_path) or '-'}\n")
        if len(main_summ):
            fh.write("\n=== Main comparison (mean over seeds) ===\n" + main_summ[cols].round(4).to_string(index=False) + "\n")
        if len(abl_summ):
            fh.write("\n=== Ablation (mean over seeds) ===\n" + abl_summ[cols].round(4).to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
