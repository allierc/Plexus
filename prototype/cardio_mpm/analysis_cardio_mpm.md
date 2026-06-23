# Cardio-MPM Loop — Running Analysis

Per-batch analysis written by the agent-in-the-loop. Each batch: the parallel `cardio_mpm_train.py`
jobs, their final interior **R²**, the dashboard read (red-on-green superposition, learned stiffness +
direction dx/dy structure), the winner, and the reasoning for the next `cardio_mpm_plan.json`. Durable
claims distilled here live in `knowledge_cardio_mpm.md`.

Seed batch 1 probes the overshoot/regime knobs around `material_directional_cardio`: amplitude
{10,15,25}, lr {2e-3,5e-3}, drag_k 60, dur0 15 — to find what closes the loop (interior R² → positive)
and whether the UNet begins structuring the stiffness + direction fields.

---

<!-- agent appends dated batch sections below -->
