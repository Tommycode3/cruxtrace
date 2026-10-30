# cruxtrace

![example timeline (synthetic)](docs/img/example_timeline_synthetic.png)

<sub>Synthetic boulder session, held-out subject. Top: dynamic acceleration $\lvert\lVert\mathbf a\rVert-g\rvert$ with detected climbs shaded. Middle: HMM posterior $P(\text{climb}\mid\mathbf x_{1:T})$. Bottom: detected vs. ground truth. The blocks between climbs are walking bouts, correctly rejected.</sub>

Climbing session tracking from the raw accelerometer of a single wrist: it finds the climbs, counts them, measures time on wall and rest, and computes per-climb pauses, move rate and smoothness. Segmentation is a sticky 2-state HMM fitted by EM on each session, and on held-out synthetic subjects it reaches event F1 0.94 against 0.59 for a threshold baseline. Every number here is synthetic, because I could not find an openly licensed wrist-IMU climbing dataset with ground truth. Accuracy on real climbers is unknown until real sessions are recorded and labelled ([`docs/DATA_COLLECTION.md`](docs/DATA_COLLECTION.md)).

<!-- TODO(Tommaso): dictate 3-4 lines in your own words -->

## Usage

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # numpy, scipy, matplotlib (+ fitdecode, pytest)

# try it on a synthetic session (clearly stamped SYNTHETIC in the report)
climbtrack synth -o s.csv --labels s_labels.csv --seed 7
climbtrack analyze s.csv --labels s_labels.csv -o out --synthetic

# your own data
climbtrack analyze my_session.csv -o out            # CSV: t, ax, ay, az [, gx, gy, gz]
climbtrack analyze my_session.fit -o out            # FIT with accelerometer_data messages
```

The CSV reader auto-detects time units (s, ms, ns), accelerometer units (g or m/s²), common column names from SensorLog and Sensor Logger, and an optional gyroscope. Jittery timestamps are resampled onto a uniform grid. Output is a Markdown/JSON report with a timeline figure. An hour of 50 Hz data takes a few seconds of CPU.

```bash
pytest -q                          # ~10 s
python scripts/bench_synthetic.py  # ~1 min, rewrites docs/bench_synthetic.md + figure
```

## How it works

A wrist accelerometer measures specific force in the sensor frame, $\mathbf a(t) = R(t)^\top\big(\ddot{\mathbf p}(t) + g\,\mathbf e_z\big) + \mathbf b + \boldsymbol\varepsilon(t)$. The norm ignores how the watch sits on the wrist, so $d(t) = \lvert\lVert\mathbf a(t)\rVert - g\rvert$ is zero at rest in any posture. It misses slow re-orientations of the arm, though, and those are most of what climbing looks like at the wrist, so gravity is estimated too: a 0.3 Hz zero-phase low-pass of $\mathbf a$, normalized. With a gyro, a complementary filter on the gravity vector (the attitude-only case of Mahony et al. 2008) gives a better $\hat{\mathbf g}$ for the movement metrics:

$$
\tilde{\mathbf g}_k = \hat{\mathbf g}_{k-1} + \hat{\mathbf g}_{k-1}\times\boldsymbol\omega_k\,\Delta t,
\qquad
\hat{\mathbf g}_k = \frac{\alpha\,\tilde{\mathbf g}_k + (1-\alpha)\,\mathbf a_k/\lVert\mathbf a_k\rVert}
{\lVert\cdot\rVert},
\qquad \alpha = \frac{\tau}{\tau+\Delta t},\ \tau = 1\ \text{s}.
$$

Segmentation always uses the accelerometer-only estimate, so it behaves the same on watches that export no gyro. Over 4 s windows with a 1 s hop it computes four features: the fraction of time the forearm is raised above 30°, the gravity tilt rate, the log RMS of $d$, and gait periodicity (the autocorrelation peak at 1.3–2.9 steps/s). A climb is a minutes-long regime and holds or chalk dips look like rest at 4 s, so the HMM sees a 15 s running median of the first three. A median, so a single gesture at rest cannot light up its neighbours.

The HMM has states rest and climb with diagonal Gaussian emissions. The transition matrix is fixed from expected durations instead of learned, and the emission log-likelihood is tempered because overlapping windows are not independent:

$$
A_{kk} = 1-\frac{\Delta}{\bar D_k},\quad \bar D_{\text{rest}} = 150\ \text{s},\ \bar D_{\text{climb}} = 60\ \text{s},
\qquad
\log \tilde p(\mathbf x_t\mid z_t=k) = \beta\,\log\mathcal N(\mathbf x_t;\boldsymbol\mu_k,\operatorname{diag}\boldsymbol\sigma_k^2),\quad \beta = \tfrac{\text{hop}}{\text{window}} = 0.25 .
$$

Emissions are fitted per session by Baum–Welch (Rabiner 1989), initialized from a sigmoid of robust z-scores, so the model adapts to each person and device without labels. Windows with gait periodicity above 0.5 are forced to rest. I chose an HMM over change-point detection such as PELT because it gives a posterior for human correction and makes the duration prior explicit. Viterbi gives the path; climbs closer than 10 s are merged and climbs shorter than 10 s dropped.

Inside each climb, a pause is at least 2 s with linear acceleration below 0.6 m/s² and rotation below 0.35 rad/s, and a move is an active run of at least 0.3 s. Smoothness per move is LDLJ-A (Melendez-Calderon et al. 2021) on linear acceleration and SPARC (Balasubramanian et al. 2012) on rotation rate:

$$
\mathrm{LDLJ\text{-}A} = -\ln\!\left(\frac{T}{a_{\text{peak}}^{2}}\int_{0}^{T}\left\lVert\frac{d\mathbf a_{\text{lin}}}{dt}\right\rVert^{2}dt\right),
\qquad
\mathrm{SPARC} = -\int_{0}^{\omega_c}\sqrt{\Big(\frac{1}{\omega_c}\Big)^{2}+\Big(\frac{d\hat V}{d\omega}\Big)^{2}}\,d\omega .
$$

Both have closed-form unit tests on a minimum-jerk reach: LDLJ-V $=-\ln(720/1.875^2)\approx-5.32$ for any distance and duration, and SPARC ≈ $-1.40$. Seifert et al. (2014) used hip jerk to separate climbing skill levels, but a wrist is not a hip, so read these as within-person trends on one device.

## Results on synthetic data

Subjects 0–3 were used while designing features and thresholds; subjects 4–15 are held out, each with one boulder and one rope session. Events match at IoU ≥ 0.5, one-to-one.

| method | fs | climbs | P | R | event F1 | count MAE | time-on-wall rel. err |
|---|---|---|---|---|---|---|---|
| HMM | 50 | 177 | 0.94 | 0.95 | 0.94 | 0.58 | 5.0 % |
| HMM | 25 | 177 | 0.95 | 0.92 | 0.93 | 0.79 | 7.8 % |
| hysteresis threshold | 50 | 177 | 0.49 | 0.76 | 0.59 | 4.46 | 17.8 % |

Full table in [`docs/bench_synthetic.md`](docs/bench_synthetic.md). This shows the code is wired correctly and that temporal context beats per-window thresholds under the generator's assumptions, nothing more. The typical error is a climb detected a few seconds early, because the generator puts fidgeting while reading the problem just before each climb.

The generator (`climbtrack.synth`) builds a 6-axis IMU from a kinematic model with minimum-jerk transitions: rest with random gestures every 12–40 s, walking at 1.6–2.1 steps/s, belaying with rope pulls at about 0.6 Hz as a deliberate confounder, and climbing as alternating moves and holds with 8–12 Hz tremor. It has no falls, down-climbing, traversing or strap slip.

## Why there is no real-data benchmark

I searched Zenodo, Figshare, UCI, PhysioNet, GitHub and papers' data statements in October 2026. ClimbSense (Kosmalla et al. 2015), ClimbAX (Ladha et al. 2013), ClimBSN (Pansiot et al. 2008) and Boulanger et al. (2016) published no data I could find. CIMI4D and AscendMotion/ClimbingCap are full-body mocap of climbing only, with no rest or walking context, no wrist-watch setup and access by request. MoonBoard-style grade prediction was skipped for lack of a clear open data licence. [`docs/DATA_COLLECTION.md`](docs/DATA_COLLECTION.md) has a recording protocol, the `climbtrack label` stopwatch labeller, `climbtrack align` for 3-tap sync, and the evaluation to run once there are at least 5 people.

## Limitations

- No real-data validation. Thresholds (30° arm-up, plausibility gate, 10 s merge) are checked on synthetic data only.
- Wrist only. "Smoothness" is hand smoothness; footwork and hips are invisible.
- Arms-up activity off the wall (stretching, brushing holds) can look like climbing.
- Attempts less than 10 s apart merge into one climb (`--merge-gap`).
- Without a gyro, orientation changes faster than 0.3 Hz leak into linear acceleration.
- FIT parsing is untested on a real device file; most watches do not store raw `accelerometer_data`.

## References

* Balasubramanian, S., Melendez-Calderon, A., Burdet, E. (2012). A robust and sensitive metric for quantifying movement smoothness. *IEEE Trans. Biomed. Eng.* 59(8). https://doi.org/10.1109/TBME.2011.2179545
* Balasubramanian, S., Melendez-Calderon, A., Roby-Brami, A., Burdet, E. (2015). On the analysis of movement smoothness. *J. NeuroEng. Rehabil.* 12:112. https://doi.org/10.1186/s12984-015-0090-9
* Boulanger, J., Seifert, L., Hérault, R., Coeurjolly, J.-F. (2016). Automatic sensor-based detection and classification of climbing activities. *IEEE Sensors J.* 16(3), 742–749. https://doi.org/10.1109/JSEN.2015.2481511
* Hogan, N., Sternad, D. (2009). Sensitivity of smoothness measures to movement duration, amplitude, and arrests. *J. Motor Behavior* 41(6). https://doi.org/10.3200/35-09-004-RC
* Killick, R., Fearnhead, P., Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *JASA* 107(500). https://doi.org/10.1080/01621459.2012.737745
* Kosmalla, F., Daiber, F., Krüger, A. (2015). ClimbSense: Automatic climbing route recognition using wrist-worn inertia measurement units. *CHI '15*. https://doi.org/10.1145/2702123.2702311
* Ladha, C., Hammerla, N. Y., Olivier, P., Plötz, T. (2013). ClimbAX: Skill assessment for climbing enthusiasts. *UbiComp '13*. https://doi.org/10.1145/2493432.2493492
* Mahony, R., Hamel, T., Pflimlin, J.-M. (2008). Nonlinear complementary filters on the special orthogonal group. *IEEE Trans. Automatic Control* 53(5). https://doi.org/10.1109/TAC.2008.923738
* Melendez-Calderon, A., Shirota, C., Balasubramanian, S. (2021). Estimating movement smoothness from inertial measurement units. *Front. Bioeng. Biotechnol.* 8:558771. https://doi.org/10.3389/fbioe.2020.558771
* Pansiot, J., King, R. C., McIlwraith, D. G., Lo, B. P. L., Yang, G.-Z. (2008). ClimBSN: Climber performance monitoring with BSN. *ISSS-MDBS 2008*. https://doi.org/10.1109/ISSMDBS.2008.4575009
* Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proc. IEEE* 77(2). https://doi.org/10.1109/5.18626
* Seifert, L., Orth, D., Boulanger, J., Dovgalecs, V., Hérault, R., Davids, K. (2014). Climbing skill and complexity of climbing wall design: assessment of jerk as a novel indicator of performance fluency. *J. Appl. Biomech.* 30(5). https://doi.org/10.1123/jab.2014-0052

MIT, see [LICENSE](LICENSE). To cite, see [CITATION.cff](CITATION.cff).
