"""Search distillation: the tree teaches the policy, game by game.

One learner, seated against the three house bots running exactly what
production runs.  At every learner decision the search measures Q for a
shortlist of candidates, and the policy is trained to *agree with those
Q values* -- the tree's conclusions become the net's instincts, and the
next batch searches from a better prior.  This is expert iteration:
each decision carries a target computed from 32 simulations rather than
a single noisy game outcome, which is why it needs orders of magnitude
fewer games than PPO.

Why Q and not visit counts.  The search allocates simulations *evenly*
across its shortlist, so visit counts are uniform by construction and
carry exactly zero information -- the AlphaZero target would be a
constant.  Q is the signal, and with 32 determinizations per candidate
it is a genuinely estimated mean rather than a bandit's noisy tail.

Three deliberate choices:

* **Candidates are sampled from the prior**, not taken as its top few.
  Top-N only ever re-measures what the net already believes, so a move
  it underrates never receives a target and the error never gets
  corrected.  Pass is always seated -- hold-or-spend is where the human
  study located the leak.
* **Nothing searches inside a rollout.**  Nested search would be
  32 x 16 x 32 evaluations per decision; opponents inside a rollout
  answer with the raw policy, sampled from its distribution.
* **The move played is sampled** from the Q target, not argmaxed, so
  2000 games explore 2000 different lines instead of replaying one.

The value head trains on the game's real outcome and the belief head on
the true hands (perfect-information supervision, free at training time).

    python -m big2.distill --games 2000 --report-every 100
    python -m big2.distill --eval-only --eval-games 1000
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from big2.game import Big2Game, ScoringConfig
from big2.rules import DEFAULT_RULES

HERE = os.path.dirname(os.path.abspath(__file__))
POLICIES = os.path.join(HERE, "policies")

START = "wangbot_v2.pt"                     # the learner's lineage
HOUSE = ("wangbot2", "khabib", "sicario")   # the production slate
OUT = os.path.join(POLICIES, "wangbot_v3.pt")
LATEST = os.path.join(POLICIES, "wangbot_v3_latest.pt")

# Training search: 32 sims per candidate, at most 5 candidates (pass
# always among them), depth 12 opening / 16 from the midgame on.
SIMS_PER_CANDIDATE = 32
MAX_CANDIDATES = 5
DEPTH_BY_PHASE = ((40, 12), (0, 16))
Q_TEMP = 0.05          # Q units, set from the MEASURED noise floor, not
#   from a decision threshold.  Re-searching the same positions across
#   seeds: between-candidate spread 0.215, across-seed noise 0.022
#   (signal/noise 9.6).  Q is normalized so 1.0 = 39 points, which makes
#   that 0.215 spread worth ~8.4 points/game -- a gap that must become a
#   decisive target, not a mild preference.  At tau = 2x the noise
#   floor, a real gap is ~74x while two moves inside noise stay within
#   1.5x: sharp where the measurement is trustworthy, flat where it is
#   not.  (An earlier 0.10 came from OVERRIDE_MARGIN_Q, which is the bar
#   for overruling the policy -- a threshold, not a temperature.)
PLAY_TEMP = 1.0        # temperature for sampling the move actually played

# Evaluation: every seat searching at production settings.
EVAL_SIMS = 16
EVAL_DEPTH = 10
EVAL_BUDGET = 0.5


def _policy(path: str):
    from big2.neural import PPOPolicy

    return PPOPolicy.load(os.path.join(POLICIES, path))


def learner_agent(policy, seed: int):
    """The training agent: sampled shortlist, per-candidate budget."""
    from big2.agent import IntegratedAgent

    return IntegratedAgent(
        policy, seed=seed, time_budget=1e6,          # sims bound it, not clock
        sims_per_candidate=SIMS_PER_CANDIDATE,
        max_candidates=MAX_CANDIDATES,
        sample_candidates=True,
        depth_by_phase=DEPTH_BY_PHASE,
    )


def eval_agent(policy, seed: int):
    """Production shape: 16 sims/candidate, depth 10, 500ms."""
    from big2.agent import IntegratedAgent

    return IntegratedAgent(
        policy, seed=seed, time_budget=EVAL_BUDGET, depth=EVAL_DEPTH,
        sims_per_candidate=EVAL_SIMS, max_candidates=MAX_CANDIDATES,
    )


# ----------------------------------------------------------------------
# Rollout worker: play games, harvest Q targets
# ----------------------------------------------------------------------


def play_games(args) -> bytes:
    (blob, n_games, seed) = args
    import torch

    torch.set_num_threads(1)
    from big2.endgame import move_key
    from big2.neural import (
        PPOPolicy, belief_target, build_net, encode_decision,
    )
    from big2.webapi import get_ai

    payload = pickle.loads(blob)
    net = build_net(**payload["arch"])
    net.load_state_dict(payload["state_dict"])
    net.eval()
    pol = PPOPolicy(net)
    # Encode exactly as this checkpoint was trained (the v2 line carries
    # the extra 'beat' action block); defaults would build a narrower row.
    enc = dict(include_profiles=pol.uses_profiles,
               include_danger=pol.uses_danger,
               include_plan=pol.uses_plan,
               include_beat=getattr(pol, "uses_beat", False))
    learner = learner_agent(pol, seed=seed)
    house = [get_ai(k, seed=seed + 11 * i) for i, k in enumerate(HOUSE)]

    rng = random.Random(seed)
    samples: List[Dict] = []
    scores: List[float] = []
    searched = forced = 0

    for _ in range(n_games):
        seat = rng.randrange(4)                    # rotate the learner's chair
        opps = list(house)
        rng.shuffle(opps)
        game = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                        num_players=4,
                        rng=random.Random(rng.randrange(2 ** 31)))
        mine: List[Dict] = []
        while not game.game_over:
            p = game.turn
            if p != seat:
                k = [q for q in range(4) if q != seat].index(p)
                game.step(opps[k].select(game, p))
                continue
            options, state, acts = encode_decision(game, p, **enc)
            if len(options) == 1:
                forced += 1
                game.step(options[0])
                continue
            dec = learner.explain(game, p)
            qs = dec.values or {}
            live = [(i, m) for i, m in enumerate(options)
                    if move_key(m) in qs and dec.visits.get(move_key(m), 0)]
            if len(live) < 2:
                # Solver or forced-shape decision: no distribution to
                # teach, but the move still gets played.
                mv = next((m for m in options if move_key(m) == dec.move),
                          options[0])
                game.step(mv)
                continue
            searched += 1
            idx = np.array([i for i, _m in live])
            q = np.array([qs[move_key(m)] for _i, m in live], dtype=np.float64)
            z = np.exp((q - q.max()) / max(Q_TEMP, 1e-6))
            target = z / z.sum()
            mine.append({
                "state": state, "acts": acts,
                "n_options": len(options),
                "cand": idx.astype(np.int64),
                "target": target.astype(np.float32),
                "belief": belief_target(game, p),
            })
            # Play a sample from the target: 2000 games should explore
            # 2000 lines, not replay the same one.
            pick = z ** (1.0 / max(PLAY_TEMP, 1e-6))
            pick = pick / pick.sum()
            chosen = live[rng.choices(range(len(live)), weights=pick)[0]][1]
            game.step(chosen)

        final = float(game.scores[seat]) / 39.0
        scores.append(float(game.scores[seat]))
        for s in mine:
            s["value"] = final
        samples.extend(mine)

    return pickle.dumps({"samples": samples, "scores": scores,
                         "searched": searched, "forced": forced})


# ----------------------------------------------------------------------
# Learner update
# ----------------------------------------------------------------------


def update(net, opt, samples, epochs: int = 2, batch: int = 64,
           value_coef: float = 0.5, belief_coef: float = 0.3):
    """Cross-entropy toward the Q target, plus value and belief heads."""
    import torch
    import torch.nn.functional as F

    if not samples:
        return {}
    order = list(range(len(samples)))
    rng = random.Random(0)
    stats = {"policy": 0.0, "value": 0.0, "belief": 0.0, "n": 0}
    net.train()
    for _ in range(epochs):
        rng.shuffle(order)
        for i in range(0, len(order), batch):
            chunk = [samples[j] for j in order[i : i + batch]]
            b = len(chunk)
            a = max(s["n_options"] for s in chunk)
            sdim = chunk[0]["state"].shape[0]
            adim = chunk[0]["acts"].shape[1]
            S = torch.zeros(b, sdim)
            A = torch.zeros(b, a, adim)
            M = torch.zeros(b, a, dtype=torch.bool)     # legal options
            C = torch.zeros(b, a, dtype=torch.bool)     # evaluated candidates
            T = torch.zeros(b, a)
            V = torch.zeros(b)
            B = torch.zeros(b, chunk[0]["belief"].shape[0])
            for k, s in enumerate(chunk):
                n = s["n_options"]
                S[k] = torch.from_numpy(s["state"])
                A[k, :n] = torch.from_numpy(s["acts"])
                M[k, :n] = True
                cand = torch.from_numpy(s["cand"])
                C[k, cand] = True
                T[k, cand] = torch.from_numpy(s["target"])
                V[k] = s["value"]
                B[k] = torch.from_numpy(s["belief"])
            logits, value, belief = net(S, A, M)
            # The target covers only the SAMPLED candidates, so the loss
            # must too.  Normalizing over every legal move would teach
            # "everything I did not evaluate is worth zero" -- a claim
            # the search never made, and one that would progressively
            # collapse the policy onto whatever the shortlist happened
            # to draw.  Restricted to the candidates, the loss teaches
            # exactly what was measured: their relative worth.
            logp = F.log_softmax(logits.masked_fill(~C, -1e9), dim=1)
            pol = -(T * logp).sum(1).mean()
            val = F.mse_loss(value, V)
            bel = F.binary_cross_entropy_with_logits(belief, B)
            loss = pol + value_coef * val + belief_coef * bel
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats["policy"] += float(pol.detach()) * b
            stats["value"] += float(val.detach()) * b
            stats["belief"] += float(bel.detach()) * b
            stats["n"] += b
    net.eval()
    n = max(1, stats["n"])
    return {k: v / n for k, v in stats.items() if k != "n"}


# ----------------------------------------------------------------------
# Evaluation: every seat searching at production settings
# ----------------------------------------------------------------------


def evaluate(path: str, n_games: int, seed: int = 5150,
             workers: Optional[int] = None) -> Dict:
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    per = max(1, n_games // workers)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        parts = pool.map(
            _eval_chunk,
            [(path, per, seed + 7919 * w) for w in range(workers)])
    total = sum(p["total"] for p in parts)
    games = sum(p["games"] for p in parts)
    wins = sum(p["wins"] for p in parts)
    return {"games": games, "score": total / max(1, games),
            "win_rate": wins / max(1, games)}


def _eval_chunk(args) -> Dict:
    path, n_games, seed = args
    import torch

    torch.set_num_threads(1)
    from big2.neural import load_checkpoint_policy
    from big2.webapi import get_ai

    torch.set_grad_enabled(False)
    cand = eval_agent(load_checkpoint_policy(path), seed=seed)
    house = [get_ai(k, seed=seed + 3 * i) for i, k in enumerate(HOUSE)]
    rng = random.Random(seed)
    total = wins = 0.0
    for g in range(n_games):
        seat = g % 4
        opps = list(house)
        rng.shuffle(opps)
        seats = []
        j = 0
        for p in range(4):
            if p == seat:
                seats.append(cand)
            else:
                seats.append(opps[j])
                j += 1
        game = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                        num_players=4,
                        rng=random.Random(rng.randrange(2 ** 31)))
        scores = game.play_out(seats)
        total += scores[seat]
        wins += 1.0 if game.winner == seat else 0.0
    return {"total": total, "games": n_games, "wins": wins}


# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", default=START)
    parser.add_argument("--resume", action="store_true",
                        help="continue from the rolling checkpoint and its "
                             "recorded game count.  This box restarts every "
                             "few hours; without resume a long run loses "
                             "everything, with it a restart costs one batch.")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-every", type=int, default=500,
                        help="mid-run checkpoint eval, in games (0 = off). "
                             "Distillation can soften a peaked policy far "
                             "faster than it improves it; this reads the "
                             "trajectory instead of the endpoint.")
    parser.add_argument("--eval-every-games", type=int, default=200)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    import torch

    if args.eval_only:
        rep = evaluate(args.out, args.eval_games, workers=args.workers)
        print(f"[eval] {rep['games']} games vs the house slate "
              f"(all seats searching, {EVAL_SIMS} sims/cand, depth "
              f"{EVAL_DEPTH}, {int(1000 * EVAL_BUDGET)}ms): "
              f"{rep['score']:+.3f}/game, win rate {rep['win_rate']:.3f}",
              flush=True)
        return

    workers = args.workers or max(1, (os.cpu_count() or 4) - 1)
    # Trust the checkpoint's own architecture (widths, MLP depth, and
    # the second attention block) rather than today's constants.
    from big2.neural import load_checkpoint_policy

    start_path = os.path.join(POLICIES, args.start)
    done = 0
    if args.resume and os.path.exists(LATEST):
        start_path = LATEST
        try:
            done = int(torch.load(LATEST, map_location="cpu",
                                  weights_only=True)
                       .get("meta", {}).get("games", 0))
        except Exception:
            done = 0
        print(f"[distill] resuming from {LATEST} at {done} games", flush=True)
    payload = torch.load(start_path, map_location="cpu", weights_only=True)
    sd = payload["state_dict"]
    arch = {
        "d_model": payload.get("d_model", 192),
        "heads": payload.get("heads", 4),
        "state_dim": sd["state_mlp.0.weight"].shape[1],
        "act_dim": sd["act_mlp.0.weight"].shape[1],
        "layers": payload.get("layers", 2),
        "attn_blocks": 2 if any(k.startswith("attn2.") for k in sd) else 1,
    }
    net = load_checkpoint_policy(start_path).net
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    print(f"[distill] learner {args.start} vs {list(HOUSE)} | "
          f"{SIMS_PER_CANDIDATE} sims/candidate, max {MAX_CANDIDATES} "
          f"(pass always seated), depth {DEPTH_BY_PHASE} | "
          f"{args.games} games, {workers} workers", flush=True)

    ctx = mp.get_context("spawn")
    rng = random.Random(args.seed + done)   # don't replay the same deals
    t0 = time.time()
    started_at = done
    hist: List[float] = []
    while done < args.games:
        want = min(args.report_every, args.games - done)
        per = max(1, want // workers)
        blob = pickle.dumps({
            "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
            "arch": arch})
        with ctx.Pool(workers) as pool:
            parts = [pickle.loads(r) for r in pool.map(
                play_games,
                [(blob, per, rng.randrange(2 ** 31)) for _ in range(workers)])]
        samples = [s for p in parts for s in p["samples"]]
        scores = [s for p in parts for s in p["scores"]]
        played = per * workers
        done += played
        stats = update(net, opt, samples, epochs=args.epochs)
        hist.extend(scores)
        rate = (done - started_at) / max(1e-9, time.time() - t0)
        recent = hist[-400:]
        print(
            f"[distill] {done}/{args.games} games | "
            f"score {np.mean(scores):+.2f} (last400 {np.mean(recent):+.2f}) | "
            f"targets {len(samples)} | "
            f"pol {stats.get('policy', 0):.3f} val {stats.get('value', 0):.3f} "
            f"bel {stats.get('belief', 0):.3f} | "
            f"{rate * 3600:.0f} games/h | "
            f"eta {(args.games - done) / max(rate, 1e-9) / 60:.0f}m",
            flush=True)
        torch.save({"state_dict": {k: v.cpu()
                                   for k, v in net.state_dict().items()},
                    "d_model": arch["d_model"], "heads": arch["heads"],
                    "layers": arch["layers"],
                    "meta": {"games": done, "note": "v3-distill"}},
                   LATEST)
        if args.eval_every and done % args.eval_every == 0 \
                and done < args.games:
            rep = evaluate(LATEST, args.eval_every_games,
                           seed=1234 + done, workers=args.workers)
            print(f"[check] @{done} games: {rep['score']:+.3f}/game vs the "
                  f"house slate ({rep['games']} games, win rate "
                  f"{rep['win_rate']:.3f})", flush=True)
    os.replace(LATEST, args.out)
    print(f"[distill] trained {done} games -> {args.out}", flush=True)

    if not args.skip_eval:
        rep = evaluate(args.out, args.eval_games, workers=args.workers)
        print(f"[eval] {rep['games']} games vs the house slate "
              f"(all seats searching): {rep['score']:+.3f}/game, "
              f"win rate {rep['win_rate']:.3f}", flush=True)


if __name__ == "__main__":
    main()
