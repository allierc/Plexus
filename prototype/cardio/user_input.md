# User input — pending (acknowledge in analysis.md, then act)

## 2026-06-23 — THE fitRMSE METRIC WAS BROKEN; honest metric added. Re-rank everything.

**Finding (verified offline on b04.s2 gfloor055, the "best" 0.00221 run):**
- The free render (`true_vs_learned.png`) does NOT match the GT — checked numerically.
- Interior render error = 0.00321 vs a **do-nothing baseline (predict zero interior motion) = 0.00064**
  → the "best" model is **5× WORSE than doing nothing**; **motion R² = −13.3**.
- Why `fitRMSE` looked tiny (and is NOT a goodness measure): (1) ~30% of nodes are the Dirichlet-pinned
  **boundary band** → pred==GT → exactly 0 error, diluting the mean; (2) the real motion is tiny (~6e-4)
  so all absolute errors are tiny; (3) it averages over near-static background interior nodes.
- A model that predicts **no interior motion** scores BETTER than every run so far. So all "best RMSE"
  rankings in `knowledge.md`/`analysis.md` are invalid as *goodness* claims (relative damping/streak
  trends may still hold; absolute "we fit the loops" is FALSE — current best R² ≈ −13).
- `b01_s0` "render" is literally **NaN** (frozen run) — never a valid result.

**Code changes already applied to `cardio_train08_09.py` (take effect from the NEXT batch you launch):**
1. `l_fit` now excludes the pinned boundary band — **interior nodes only** (`rmse_m(..., interior)`).
2. New HONEST metrics logged to `progress.txt`, stdout, and `run.log`:
   - `fitRMSE_int` — interior-only RMSE.
   - **`R2`** — motion-normalised variance-explained over **interior AND moving** nodes. **>0 good,
     1.0 = perfect, R2<0 = worse than predicting no motion.** This is the metric to RANK on.
   - `NRMSE` = sqrt(1−R2) = RMSE / GT-motion-RMS (1.0 = no skill).
   - `run.log` gets a `render_fit interior_moving: R2=… NRMSE=…` line computed on the FREE render
     (what the image shows); NaN `fit_pos` → `nan` so you can detect/skip frozen runs.

**Directives for the next batch:**
1. **RANK on `R2` (free-render `render_fit R2` in run.log), not fitRMSE.** Treat `R2 ≤ 0` as a failed
   fit regardless of how small fitRMSE is. Skip any run with `R2=nan` (NaN render) — flag as FAILED.
2. **Re-frame the program goal:** the loops OPEN (real qualitative win) but **overshoot ~5×** and are
   mis-directed — they are not calibrated to the tiny real motion. The interesting open question is now
   **"can the open loops be CALIBRATED to the real loops (R2 → positive)?"**, not "lowest fitRMSE".
3. **Likely root cause = under-constrained interior.** Excluding the boundary from `l_fit` is step 1, but
   the static-background interior nodes still dilute it. Strongly consider making `l_fit` itself
   **motion-normalised on the moving-node mask** (reuse `fit_mask`/`open_mask`) so interior overshoot is
   actually penalised during training — this could *calibrate* the loop amplitude, not just measure it.
   Propose this as a batch-6 slot (one-knob: a `CARDIO_FIT_NORM=1` flag) with the current `l_fit` as the
   ablation control.
4. Add a `knowledge.md` Established/Open entry recording that **fitRMSE is boundary-diluted and not a
   goodness measure; R2 is the metric**, and re-label the comparison-table "best" rows accordingly
   (note the absolute fits are poor, R2<0 to date).
