# User input

Written by Cedric, read fresh by the Proposer and the Analyst at the top of every round.
Edit this file mid-campaign -- the next round picks it up, no relaunch needed.

## Pending Instructions

### 1. Three of your own conclusions were about broken instruments, not about biology. Retract them.

Between round 7 and now, three defects were found BELOW the level you can edit. Each one made an
operator silently do nothing, so your record contains "this mechanism does not work" where the
truth is "this mechanism never ran". Please do not carry these forward:

- **`rd_interface_tension` (the `extrude` node) never acted in ANY run of this campaign.** The
  acted-ledger reads `rd_interface_tension: 0` across 800 scheduled frames. Its threshold `a_sw`
  was an absolute value on the activator, declared over (0.2, 6.0) and defaulting to 0.5, while
  the activator's own maximum across 78 runs has a MEDIAN OF 0.000 and a ceiling of 1.541 -- so
  it selected zero cells and returned early every frame. `a_sw` is now a FRACTION of the
  activator's own maximum (0.6 = "the top 40% of the field is red"), so it fires at any operating
  point. Your rounds 6 and 7 both reported it "inert" and you were right; the reason was not the
  mechanism. **It is untested, not refuted.**

- **`divide_3d` with `orient_iface` was behaviourally identical to `hertwig` in 74% of runs.** Its
  gate `orient_asw` had the same absolute (0.2, 6.0) range, defaulting to 1.0, and only 20 of 78
  runs ever reached `act_max > 1.0`. So a `set_impl divide_3d orient_iface` edit built, ran,
  recorded -- and was not a mechanism edit. Also now a fraction of the field. **Oriented division
  is untested.**

- **The "rail/copy, launcher duplication" you have reported since round 5 is not the launcher.**
  `r007_00_ctrl` and `r007_10` differ in exactly one line -- `n_spots: 1` vs `2` -- and are
  bit-identical because `n_spots` is read only by `mode: cones`, and every parent runs
  `mode: scatter`. A dead knob, now withheld from the menu. Do not spend further slots
  diagnosing the launcher.

### 2. What I would like you to spend this campaign on.

Your own finding is that coupling is settled: `corr_act_rad_peak` 0.852, replicated at n=3, and
"pattern grips a growing shape, robustly, and still makes only bulges". I agree, and I agree with
your instruction to yourself not to re-propose coupling retunes. Two things follow.

**The open question is elongation, not grip.** `protr_peak` has been 1.16-1.22 for seven rounds
against the >=1.3 a tube needs, and you called the wall real rather than a tuning gap. The spot
field is now right (~10 spots ~10 cells apart, Fig. 5a). So the missing ingredient is whatever
turns a lobe into a finger, and the two mechanisms that could supply it -- interface line tension
and oriented division -- are the two that were never actually running.

**Test the purse-string separately from the push.** `rd_interface_tension` carries two terms:

    E = K_purse * sum_iface(edge length)  -  K_extrude * sum_red(a * r)

`K_purse` is a line tension on the red/white boundary -- ordinary vertex-model physics, and how a
real purse-string works. `K_extrude` is an energy that FALLS as red cells move outward: it is not
a mechanism, it is the answer written into the objective, and a run carrying it can only ever be
a control. `K_purse` was hard-wired to 0 until now and is a free parameter for the first time.

So: **`K_extrude = 0` with `K_purse > 0` is the experiment I most want to see.** If an interface
line tension alone necks a lobe into a finger, that is a result. A run with `K_extrude > 0` is
worth ONE slot as a positive control -- to show the instruments can detect a tube when one exists
-- and its `mech_p_ratio` (~3 = forced, ~1 = grown) must be reported whenever you discuss it. A
forced run will never be promoted to a parent; that is enforced, not a request.

### 3. Two standing rules.

- **Say when a mechanism is untested rather than refuted.** The three items above cost this
  campaign real slots because a broken operator and a false hypothesis were recorded identically.
  If an operator's acted-count is 0, that run is not evidence about the mechanism.
- **Growth is not optional.** Okuda's figures go 2032 -> 2843 -> 3572 cells. Your sixth
  confirmation says growth dilutes coupling; that is a finding about a trade-off, not a licence to
  stop growing. A morphology on a shell of fixed cell count cannot be the paper's.

## Acknowledged
