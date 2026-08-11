"""A small decoder-only transformer in NumPy, with hand-written backward passes.

Why NumPy rather than a framework: this system's central claims are about
reproducibility -- "resume produces exactly the expected batch", "a fork diverges
only after the fork point". Those proofs are only as trustworthy as the
determinism underneath them. A single-threaded NumPy implementation with an
explicit operation order is bit-exact across runs on the same machine, with no
autograd nondeterminism, no kernel selection heuristics and no version drift.

The gradients are verified against central finite differences in
tests/test_model_gradients.py, so "hand-written" does not mean "unchecked".

The model is deliberately tiny (2 layers, d_model 64). The assignment is about
the data plane; the model only has to learn enough that OPUS's loss signal is
real and per-document losses are meaningful.
"""
from __future__ import annotations

import numpy as np

from .packing import IGNORE_INDEX

MAX_POSITIONS = 512


# ---------------------------------------------------------------- primitives
def layernorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = xc * inv
    return xhat * gamma + beta, (xhat, inv, gamma)


def layernorm_backward(dout, cache):
    xhat, inv, gamma = cache
    D = xhat.shape[-1]
    dgamma = (dout * xhat).reshape(-1, D).sum(0)
    dbeta = dout.reshape(-1, D).sum(0)
    dxhat = dout * gamma
    dx = (dxhat - dxhat.mean(-1, keepdims=True)
          - xhat * (dxhat * xhat).mean(-1, keepdims=True)) * inv
    return dx, dgamma, dbeta


def softmax_lastdim(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


class Transformer:
    def __init__(self, cfg, vocab_size: int):
        self.cfg = cfg
        self.V = vocab_size
        self.D = cfg.d_model
        self.H = cfg.n_heads
        self.dh = self.D // self.H
        assert self.D % self.H == 0, "d_model must divide by n_heads"
        rng = np.random.default_rng(cfg.seed)
        s = cfg.init_std

        def randn(*shape):
            return (rng.standard_normal(shape) * s).astype(np.float64)

        self.p: dict = {
            "Wte": randn(self.V, self.D),
            "Wpe": randn(MAX_POSITIONS, self.D),
            "Whead": randn(self.D, self.V),
            "ln_f_g": np.ones(self.D), "ln_f_b": np.zeros(self.D),
        }
        for i in range(cfg.n_layers):
            self.p[f"l{i}_ln1_g"] = np.ones(self.D)
            self.p[f"l{i}_ln1_b"] = np.zeros(self.D)
            self.p[f"l{i}_Wq"] = randn(self.D, self.D)
            self.p[f"l{i}_Wk"] = randn(self.D, self.D)
            self.p[f"l{i}_Wv"] = randn(self.D, self.D)
            self.p[f"l{i}_Wo"] = randn(self.D, self.D)
            self.p[f"l{i}_ln2_g"] = np.ones(self.D)
            self.p[f"l{i}_ln2_b"] = np.zeros(self.D)
            self.p[f"l{i}_W1"] = randn(self.D, cfg.d_ff)
            self.p[f"l{i}_b1"] = np.zeros(cfg.d_ff)
            self.p[f"l{i}_W2"] = randn(cfg.d_ff, self.D)
            self.p[f"l{i}_b2"] = np.zeros(self.D)

    # ------------------------------------------------------------ forward
    def forward(self, input_ids, position_ids, segment_ids, labels=None):
        B, L = input_ids.shape
        cache: dict = {"input_ids": input_ids, "position_ids": position_ids, "B": B, "L": L}

        x = self.p["Wte"][input_ids] + self.p["Wpe"][position_ids]
        cache["blocks"] = []

        # attention is causal AND intra-segment; padding attends to nothing.
        causal = np.tril(np.ones((L, L), dtype=bool))
        same = segment_ids[:, :, None] == segment_ids[:, None, :]
        real = segment_ids > 0
        allow = causal[None] & same & real[:, :, None] & real[:, None, :]
        # a fully-masked row (a pad slot) would make softmax undefined; let such
        # rows attend to themselves and discard the output via the loss mask.
        dead = ~allow.any(-1)
        allow = allow.copy()
        idx = np.arange(L)
        allow[dead, 0] = True  # harmless: those rows never bear loss
        attn_bias = np.where(allow[:, None, :, :], 0.0, -1e30)
        cache["allow"] = allow

        for i in range(self.cfg.n_layers):
            bc: dict = {}
            h1, ln1c = layernorm_forward(x, self.p[f"l{i}_ln1_g"], self.p[f"l{i}_ln1_b"])
            bc["ln1"] = ln1c
            bc["h1"] = h1

            q = h1 @ self.p[f"l{i}_Wq"]
            k = h1 @ self.p[f"l{i}_Wk"]
            v = h1 @ self.p[f"l{i}_Wv"]
            qh = q.reshape(B, L, self.H, self.dh).transpose(0, 2, 1, 3)
            kh = k.reshape(B, L, self.H, self.dh).transpose(0, 2, 1, 3)
            vh = v.reshape(B, L, self.H, self.dh).transpose(0, 2, 1, 3)
            scores = (qh @ kh.transpose(0, 1, 3, 2)) / np.sqrt(self.dh) + attn_bias
            probs = softmax_lastdim(scores)
            ctx = probs @ vh                                   # [B,H,L,dh]
            ctxm = ctx.transpose(0, 2, 1, 3).reshape(B, L, self.D)
            a = ctxm @ self.p[f"l{i}_Wo"]
            bc.update({"qh": qh, "kh": kh, "vh": vh, "probs": probs, "ctxm": ctxm})

            x_mid = x + a
            h2, ln2c = layernorm_forward(x_mid, self.p[f"l{i}_ln2_g"], self.p[f"l{i}_ln2_b"])
            z1 = h2 @ self.p[f"l{i}_W1"] + self.p[f"l{i}_b1"]
            r1 = np.maximum(z1, 0.0)
            f = r1 @ self.p[f"l{i}_W2"] + self.p[f"l{i}_b2"]
            bc.update({"ln2": ln2c, "h2": h2, "z1": z1, "r1": r1, "x_in": x, "x_mid": x_mid})
            x = x_mid + f
            cache["blocks"].append(bc)

        hf, lnfc = layernorm_forward(x, self.p["ln_f_g"], self.p["ln_f_b"])
        logits = hf @ self.p["Whead"]
        cache.update({"hf": hf, "ln_f": lnfc, "x_final": x})

        if labels is None:
            return logits, None, cache

        valid = labels != IGNORE_INDEX
        n_valid = int(valid.sum())
        flat_logits = logits.reshape(-1, self.V)
        flat_labels = labels.reshape(-1)
        flat_valid = valid.reshape(-1)

        shifted = flat_logits - flat_logits.max(-1, keepdims=True)
        logsumexp = np.log(np.exp(shifted).sum(-1))
        picked = np.where(flat_valid, flat_labels, 0)
        chosen = shifted[np.arange(flat_logits.shape[0]), picked]
        per_pos = np.where(flat_valid, logsumexp - chosen, 0.0)
        loss = per_pos.sum() / max(n_valid, 1)

        cache.update({"labels": labels, "valid": valid, "n_valid": n_valid,
                      "shifted": shifted, "logsumexp": logsumexp})
        return logits, {"loss": float(loss), "per_pos_loss": per_pos.reshape(B, L),
                        "n_valid": n_valid}, cache

    # ----------------------------------------------------------- backward
    def backward(self, cache):
        B, L = cache["B"], cache["L"]
        n_valid = max(cache["n_valid"], 1)
        g: dict = {k: np.zeros_like(v) for k, v in self.p.items()}

        probs_out = np.exp(cache["shifted"] - cache["logsumexp"][:, None])
        flat_labels = cache["labels"].reshape(-1)
        flat_valid = cache["valid"].reshape(-1)
        dlogits = probs_out
        dlogits[np.arange(dlogits.shape[0]), np.where(flat_valid, flat_labels, 0)] -= 1.0
        dlogits *= (flat_valid[:, None] / n_valid)
        dlogits = dlogits.reshape(B, L, self.V)

        hf = cache["hf"]
        g["Whead"] += hf.reshape(-1, self.D).T @ dlogits.reshape(-1, self.V)
        dhf = dlogits @ self.p["Whead"].T
        dx, dg, db = layernorm_backward(dhf, cache["ln_f"])
        g["ln_f_g"] += dg
        g["ln_f_b"] += db

        for i in reversed(range(self.cfg.n_layers)):
            bc = cache["blocks"][i]
            # ---- MLP residual
            dx_mid = dx.copy()
            df = dx
            g[f"l{i}_W2"] += bc["r1"].reshape(-1, self.cfg.d_ff).T @ df.reshape(-1, self.D)
            g[f"l{i}_b2"] += df.reshape(-1, self.D).sum(0)
            dr1 = df @ self.p[f"l{i}_W2"].T
            dz1 = dr1 * (bc["z1"] > 0)
            g[f"l{i}_W1"] += bc["h2"].reshape(-1, self.D).T @ dz1.reshape(-1, self.cfg.d_ff)
            g[f"l{i}_b1"] += dz1.reshape(-1, self.cfg.d_ff).sum(0)
            dh2 = dz1 @ self.p[f"l{i}_W1"].T
            d_from_ln2, dg2, db2 = layernorm_backward(dh2, bc["ln2"])
            g[f"l{i}_ln2_g"] += dg2
            g[f"l{i}_ln2_b"] += db2
            dx_mid = dx_mid + d_from_ln2

            # ---- attention residual
            dx_in = dx_mid.copy()
            da = dx_mid
            g[f"l{i}_Wo"] += bc["ctxm"].reshape(-1, self.D).T @ da.reshape(-1, self.D)
            dctxm = da @ self.p[f"l{i}_Wo"].T
            dctx = dctxm.reshape(B, L, self.H, self.dh).transpose(0, 2, 1, 3)

            probs = bc["probs"]
            dprobs = dctx @ bc["vh"].transpose(0, 1, 3, 2)
            dvh = probs.transpose(0, 1, 3, 2) @ dctx
            dscores = probs * (dprobs - (dprobs * probs).sum(-1, keepdims=True))
            dscores /= np.sqrt(self.dh)
            dqh = dscores @ bc["kh"]
            dkh = dscores.transpose(0, 1, 3, 2) @ bc["qh"]

            def merge(t):
                return t.transpose(0, 2, 1, 3).reshape(B, L, self.D)

            dq, dk, dv = merge(dqh), merge(dkh), merge(dvh)
            h1f = bc["h1"].reshape(-1, self.D)
            g[f"l{i}_Wq"] += h1f.T @ dq.reshape(-1, self.D)
            g[f"l{i}_Wk"] += h1f.T @ dk.reshape(-1, self.D)
            g[f"l{i}_Wv"] += h1f.T @ dv.reshape(-1, self.D)
            dh1 = (dq @ self.p[f"l{i}_Wq"].T + dk @ self.p[f"l{i}_Wk"].T
                   + dv @ self.p[f"l{i}_Wv"].T)
            d_from_ln1, dg1, db1 = layernorm_backward(dh1, bc["ln1"])
            g[f"l{i}_ln1_g"] += dg1
            g[f"l{i}_ln1_b"] += db1
            dx = dx_in + d_from_ln1

        np.add.at(g["Wte"], cache["input_ids"], dx)
        np.add.at(g["Wpe"], cache["position_ids"], dx)
        return g

    # -------------------------------------------------------------- state
    def state_dict(self) -> dict:
        return {k: v.copy() for k, v in self.p.items()}

    def load_state_dict(self, sd: dict):
        for k, v in sd.items():
            self.p[k] = np.array(v, dtype=np.float64)

    def n_params(self) -> int:
        return int(sum(v.size for v in self.p.values()))


class Adam:
    def __init__(self, params: dict, cfg):
        self.cfg = cfg
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def lr_at(self, step: int) -> float:
        if self.cfg.warmup_steps > 0 and step < self.cfg.warmup_steps:
            return self.cfg.lr * (step + 1) / self.cfg.warmup_steps
        return self.cfg.lr

    def step(self, params: dict, grads: dict, step: int) -> dict:
        self.t += 1
        c = self.cfg
        total_sq = sum(float((grads[k] ** 2).sum()) for k in grads)
        gnorm = float(np.sqrt(total_sq))
        scale = 1.0
        if c.grad_clip > 0 and gnorm > c.grad_clip:
            scale = c.grad_clip / (gnorm + 1e-12)
        lr = self.lr_at(step)
        for k in params:
            gk = grads[k] * scale
            self.m[k] = c.beta1 * self.m[k] + (1 - c.beta1) * gk
            self.v[k] = c.beta2 * self.v[k] + (1 - c.beta2) * (gk * gk)
            mhat = self.m[k] / (1 - c.beta1 ** self.t)
            vhat = self.v[k] / (1 - c.beta2 ** self.t)
            upd = lr * mhat / (np.sqrt(vhat) + c.eps)
            if c.weight_decay:
                upd = upd + lr * c.weight_decay * params[k]
            params[k] -= upd
        return {"grad_norm": gnorm, "clip_scale": scale, "lr": lr}

    def state_dict(self) -> dict:
        return {"t": self.t,
                "m": {k: v.copy() for k, v in self.m.items()},
                "v": {k: v.copy() for k, v in self.v.items()}}

    def load_state_dict(self, sd: dict):
        self.t = int(sd["t"])
        self.m = {k: np.array(v, dtype=np.float64) for k, v in sd["m"].items()}
        self.v = {k: np.array(v, dtype=np.float64) for k, v in sd["v"].items()}
