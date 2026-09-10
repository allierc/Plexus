# Seal break — the HCM sheet

The diseased (HCM) specimen was sealed by content in `discovery_cardio_mpm/_data/split.json` as
that loop's one-shot test. On 2026-09-10 Cedric authorised opening it for this prototype
("do 2 and 3 today and run the HCM fit with E fixed, which yields g, φ and δ maps for both sheets
without waiting for the E sweep"). The split file itself is not modified (its sha256 still checks);
this prototype reads `Cardio_1/1_HCM_15kPa_MR44_W3_1_MMStack_Pos0.ome.tif.derivatives.npy`
directly through `recording.load(specimen="hcm")`.

Before opening: `hcm_referee.py` (pixels vs each tracking, rest frame 110 → peak frame 67,
400 nodes): T1 = the file above r 0.997 / 0.983, 0.86 px median error on 6.8 px of motion;
T2 = `Cardio_0/derivatives.npy` identical to T1 (same measurement, stored [t, y, x] too);
T3 = `diseased.npy` r 0.30 / −0.01 — broken, like `healthy.npy`. Only T1 is used.

Recording: 299 frames, speed-peak onsets 17, 61, 121, 181, 241 (gaps 44, 60, 60, 60); mean node
displacement 2.76 px against 1.7 px for the healthy sheet.
