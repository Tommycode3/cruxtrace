# Data collection protocol (real-data benchmark: TODO)

Goal: a small, honest, labelled dataset that replaces the synthetic verifier. Target: at least
**5 climbers × 2 sessions** (one boulder, one rope if possible), so that leave-one-subject-out is
meaningful.

## 1. Recording

| item | setting |
|---|---|
| sensor | wrist IMU on the **dominant** wrist, strap snug (no slip) |
| rate | ≥ 25 Hz accelerometer; 50 Hz + gyroscope if available |
| apps | Apple Watch: any raw-sensor logger exporting CSV · Android/Wear OS: e.g. a sensor-logger app · Garmin: a Connect IQ raw-data logger that writes `accelerometer_data` into the FIT · phone strapped to the forearm also works for development |
| notes | record which wrist, which way the crown points, device model, firmware |

Export to CSV with columns `t, ax, ay, az[, gx, gy, gz]` (any of the column names in `climbtrack/io.py`).

## 2. Synchronisation (3-tap)

1. Start the IMU recording.
2. On a phone or laptop run `climbtrack label session01_live.csv`.
3. **Tap the watch face hard 3 times** (≈ 0.3 s apart) with the other hand, and type `s` + Enter at the first tap.

`climbtrack align imu.csv session01_live.csv -o session01_labels.csv` finds the 3 spikes
(> 2 g dynamic, within 3 s, first 5 min) and shifts the labels onto the IMU clock. If the taps are not
found, pass `--offset SECONDS`.

## 3. What counts as a climb (label definition)

* **start**: the last foot leaves the ground / mat.
* **end**: the first foot back on the ground (top-out walk-off, jump-off, fall, or lower-off reaching the floor).
* A fall followed by an immediate restart with < 10 s on the ground: label as **two** climbs (the
  evaluator will show these as merge errors; that is informative).
* Belaying, brushing holds, stretching, campus/hangboard: **not** climbing. Write them as free-text
  notes if they are long (a later version may label them).

Live labelling: a partner presses Enter at start/end (`u`+Enter to undo). Better: film the session
and label from video afterwards, writing `start_s,end_s,label` by hand relative to the 3 taps
(which are visible on video).

## 4. File layout

```
data/
  S01/2026-11-03_boulder/{imu.csv, labels.csv, notes.md}
  S01/2026-11-07_rope/...
  S02/...
```

`data/raw/` is git-ignored. Only share data with explicit consent; strip GPS/timestamps if publishing.

## 5. Evaluation to run

```bash
climbtrack analyze data/S01/.../imu.csv --labels data/S01/.../labels.csv -o out/S01_a
```

Report per subject and pooled: event precision/recall/F1 at IoU ≥ 0.5, mean IoU of matched events,
count MAE, exact-count rate, frame F1, time-on-wall relative error, start/end MAE. **Do not tune any
threshold on the subject being evaluated** (leave-one-subject-out: tune on the others, freeze, test).
Report failures with the timeline figure; they are the most useful result.
