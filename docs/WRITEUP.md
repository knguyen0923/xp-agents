# XP — short write-up: when does uncertainty-driven exploration pay off?

## Question
Two agents share a learner and differ by one term. **Kairi** is model-free and reward-driven (epsilon-greedy in a bandit, plain PPO in MiniGrid). **Kaiser** is Kairi plus an explicit
uncertainty-driven exploration term (discounted UCB in the bandit, PPO + an RND bonus in MiniGrid), the principle of optimism in the face of uncertainty. We ask where each wins and
where each breaks as the world moves away from the conditions both were tuned on. Kairi winning some regimes is an expected, valid result.

## Protocol
Tune each agent on the familiar knob value with tune seeds only, freeze, evaluate at every knob value on disjoint eval seeds, and report degradation curves, paired differences and
crossovers with 95% bootstrap CIs (resampling the independent seed). Both agents get equal-sized tuning grids. A tuned optimum on a grid edge makes a run provisional, and provisional
results are never reported; we widened grids and reran when it happened (reversed-tuning and drift shifts). Runs are resumable and keyed by config hash.

## Results
**Changing bandit (10 arms, sigma 1, horizon 2000, tuned at jump rate 0.001).** Kaiser beats Kairi at all seven jump rates (regret per step 0.033 to 0.072 lower, every CI excludes 0).
The advantage holds when tuning at the high jump rate and across reward noise 0.25 to 4. Kairi wins in two shifts far from the tuned point with frozen settings: 50 arms (by 0.054 [0.037, 0.071]) and arm drift 0.1 per step (by 0.209 [0.132, 0.283]), crossovers between 20 and 50 arms and between
drift 0.02 and 0.1. A post-hoc ablation retuning both agents at those points reverses both: Kaiser then wins by 0.086 [0.070, 0.103] at 50 arms and 0.168 [0.156, 0.179] at drift 0.1, and a regret-over-time split
shows Kaiser's frozen loss at 50 arms accrues after step 500, not from initial exploration. So the Kairi wins are Kaiser's tuned settings being brittle, not exploration hurting. Tag-team agents that switch between the
two help mainly where the single agents fail; `tt_meta` is the most reliable.

**MiniGrid DoorKey (tuned on 6x6, tested at 5, 6, 8, 12, 10 seeds).** With tuned settings Kairi is at or above Kaiser everywhere, and the gap excludes 0 at sizes 8 and 12. That gap is not robust:
Kaiser at a different tuning-tied RND coefficient, or at Kairi's entropy coefficient, is statistically indistinguishable from Kairi at every size. The supported conclusion is no measurable benefit
from RND on DoorKey, not that it hurts. One Kaiser seed failed outright in two of three RND settings, so seed-level failures matter. A post-hoc MultiRoom spike (28 runs, 2 seeds, 2M steps) on an exploration-heavy family found no RND benefit at that budget, but a follow-up probe at 5M steps on the borderline level (4 small rooms, 5 seeds) found that RND at coefficient 0.1 reaches 95% training success in a median 1.44M steps against 2.66M for plain PPO, faster on all 5 seeds (mean 1.17M fewer steps, CI [0.53M, 1.75M]); coefficients 0.01 and 0.03 gave no benefit. This is post-hoc (best of three coefficients, top of the range tried, tune-pool seeds), so it is evidence, not a protocol result. A protocol-legal confirmation at that level (coefficients tuned on 3 tune seeds, Kaiser's entropy matched to Kairi's, 10 disjoint eval seeds, S = 2.94M steps) then found Kaiser (`rnd_coef` 0.6) faster on all 10 seeds: mean 0.70M against 2.23M steps to 95% (paired CI of the difference [−2.12M, −0.93M]), and 10/10 seeds solved against 5/10 for Kairi. One level only; the level and grid were chosen after the probe; the step and AUC gaps are the robust numbers because Kairi's failures depend on the budget.

## Map: where each agent wins and breaks
| regime | winner |
|---|---|
| bandit, jump rate 0 to 0.02, noise 0.25 to 4, up to 20 arms, drift up to 0.02 | Kaiser |
| bandit, 50 arms (5x tuned), drift 0.1 (10x tuned), settings frozen | Kairi |
| same two points with both agents retuned there (post-hoc) | Kaiser |
| MiniGrid DoorKey 5 to 12 | tie (RND no measurable benefit) |
| MiniGrid MultiRoom, 2M steps, 2 seeds | PPO no worse; RND never helped |
| MiniGrid MultiRoom 4 small rooms, 5M steps, 5 seeds (post-hoc) | RND at coef 0.1 (about 2x fewer steps); no benefit at 0.01 or 0.03 |
| MiniGrid MultiRoom 4 small rooms, full protocol, 10 eval seeds | Kaiser (rnd 0.6): 1.5M fewer steps to 95%, faster on 10/10 seeds |
| Crafter (spike only, plain PPO, 1 seed) | not run: no area met the pre-set usable-range gate at 1M steps |

## Limitations
One bandit family and horizon; agents are frozen after tuning once (no per-shift retuning); no multiplicity correction across knob values; the ablations that explain the Kairi wins are post-hoc, cover only those two shifts, and use oracle retuning the protocol does not allow. MiniGrid: one familiar size, a 2M-step budget that missed the 80% calibration target, 10 eval seeds, and Kaiser's entropy
coefficient was fixed (not tuned) in the main run, which the matched-entropy check addresses post-hoc. See `docs/FINDINGS.md`.

## Future work
The MultiRoom confirmation at other room counts, and a deep-RL tag-team agent. A Crafter spike (plain PPO, 1M steps, one seed per area) reached 4.5 and 4.4 achievements per episode against a pre-set gate of 5, so
the Crafter phase is on hold pending a decision (`docs/spikes/crafter_spike.md`). Showdown is specified but not approved.
