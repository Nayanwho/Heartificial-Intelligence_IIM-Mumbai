# Evaluation protocol (version 2)

**Frozen on 23 October 2026, before any version-2 experiment was run.** It replaces the version-1 evaluation of 19 October, whose results are kept in `results/legacy_2026-10-19/` and labelled superseded.

**Why version 2.** An independent audit found that version 1 let information from after a forecast's origin into the inputs:

- each person's sensor offset used fingersticks from the whole period;
- the population offset prior included the patients being tested;
- missing meals were filled with a whole-period median;
- the calibration length depended on the full period length;
- sensor-off mode started from a CGM reading it should not see.

Version 2 removes each of these, and adds fair baselines and a single alert policy.

**Honest caveat.** The same 100 patients have been inspected before, in version 1 and in the audit. Version-2 results are therefore retrospective development evidence, not an independent validation. Nothing below was tuned on version-2 outcomes. The only quantities chosen after inspecting data are the calibration length and the eligibility rules, and both were chosen from data availability (period lengths and sampling), not from outcomes.

## 1. Data

- ShanghaiT2DM (Scientific Data, 2023; CC BY 4.0): 100 adults with Type 2 diabetes, 109 CGM periods, readings every 15 minutes, fingersticks, meals, insulin and oral-drug records, and a summary sheet of clinical fields.
- All periods of one patient are always kept together.
- CGM timestamps lie on a regular 15-minute grid within each period, with 10 gaps longer than 20 minutes in the whole dataset. Each period is placed on its own regular 15-minute grid; missing readings stay missing.

## 2. Timeline of one period

| Interval | Definition | Use |
| --- | --- | --- |
| Warm-up | First 24 hours of the period | Discarded everywhere |
| Calibration C | The next 5 days: [warm-up end, warm-up end + 5 days) | The only data used to personalise: sensor offset, meal fallback, physiology-core fit, typical-day profile, baseline drug flags |
| Test T | From the end of C to the last reading | Forecast origins, events and scoring |

**Eligibility.** A period is evaluated only if all three hold:

1. it has at least 1 day of CGM after the end of C;
2. it has at least 300 CGM readings in C;
3. the physiology core can be fitted on C (at least 50 usable steps).

Excluded periods are listed with the reason in `results/exclusions.csv`. The calibration length is fixed: it never depends on the length of the period or on later data.

## 3. Sensor reference and correction

- **Pairs.** Fingersticks timed inside C are matched to the nearest CGM reading inside C, within 8 minutes. Both timestamps of a pair must lie inside C.
- **Population prior.** The median (fingerstick minus CGM) over the calibration pairs of the **outer-training patients only**, recomputed for every outer fold.
- **Person offset.** (n × person median + 10 × prior) / (n + 10), where n is the person's number of calibration pairs. It is frozen for the whole of T and never updated retroactively.
- **Primary reference ("corrected").** Raw CGM plus the frozen calibration offset. This is the signal the deployed twin would see. Features, the physiology fit, events and scoring all use it, for every method.
- **Secondary reference ("raw").** Raw CGM everywhere, with no correction. The whole forecast and alarm evaluation is repeated on it and reported separately.
- **What fingersticks can and cannot show.** Sparse fingersticks describe the sensor's bias. They cannot independently confirm every low.

## 4. Forecast origins, features and targets

**Origins.** Every grid time t in T that has a reading at t and a reading 15 minutes earlier.

**Features at origin t** use only data timestamped at or before t, plus the calibration state:

- **CGM history:** the reading at t, and the readings at t − 15, t − 30, t − 60 and t − 120 minutes on the grid. A missing slot is missing; it is never replaced by another sample. Also the standard deviation of the readings in [t − 120, t].
- **Time of day:** sine and cosine of the hour.
- **Logged inputs at t:** insulin action, insulin on board, sulfonylurea and GLP-1 exposure, carbohydrate absorbing and carbohydrate on board. These come from events logged at or before t. The action curves are zero before each event.
- **Unweighed or unlogged meals:** get the person's median carbohydrate from meals logged inside C (at least 3 meals), otherwise a fixed 45 g.
- **Calibration offset.**
- **Static clinical record (EHR):** age, BMI, HbA1c, eGFR, diabetes duration, fasting C-peptide, sex, on a sulfonylurea, on insulin.
  - The summary-sheet fields carry no timestamps. They are treated as the baseline record available at the start of C; this is an assumption.
  - The drug flags use drug records timestamped before the end of C, plus the summary sheet's agent list (the same baseline assumption).
- **Physiology core:** forecasts of the change at +30, +60 and +120 minutes and the minimum over 120 minutes. These come from the person's core fitted on C, simulated on the regular grid from the reading at t, using only events logged at or before t. Future meals are unknown to every method.

**Targets.** The reference reading at exactly t + 30, t + 60 or t + 120 minutes on the grid. A missing target removes that origin from that horizon's scoring for every method.

## 5. Patient folds

- Patients are sorted by ID and shuffled with seed 7, then split into 5 outer folds of near-equal size.
- The assignment is saved in `results/folds.json` before any model is fitted.
- Every population quantity is fitted on outer-training patients only: the offset prior, the learning models, thresholds and inner-fold selections.
- Inner folds, used only to choose alert thresholds for learned scores, are 4 patient folds inside the outer-training set, with the same isolation.

The physiology core of a held-out patient is fitted on that patient's own calibration interval C. This is **personalisation after a 5-day calibration**, not a zero-shot prediction for a completely unseen person.

## 6. Forecasting methods

All methods are scored on the same origins, targets and reference.

1. **Persistence:** the reading at t.
2. **Trend:** least-squares line through the readings in [t − 30, t], at least 3 readings, extrapolated; persistence when fewer readings exist.
3. **Physiology core only.**
4. **Learning arm A, dynamic:** CGM history, time, logged inputs, calibration offset.
5. **Learning arm B, dynamic + EHR:** A plus the static clinical record. This is the "machine learning only" comparator.
6. **Learning arm D, dynamic + physiology:** A plus the core's forecasts.
7. **Learning arm C, hybrid:** A plus EHR plus physiology. This is the primary model.

**LightGBM settings, fixed and not tuned:** learning rate 0.05, 31 leaves, at least 50 rows per leaf, 0.8 row and column sampling, 400 trees, seed 7. Targets are the change from the reading at t.

**Metrics:**
- RMSE, MARD and Clarke A+B, with 95% intervals from 1,000 patient-level bootstrap resamples.
- Paired differences between methods, with patient-level bootstrap intervals: C minus B (value of physiology), C minus D and B minus A (value of the EHR), and every method minus persistence.

## 7. Low events

A CGM-defined low episode is at least 2 consecutive reference readings below 70 mg/dL, with no gap over 20 minutes. Its start is the first reading below 70. These are **sensor-defined episodes, not severe clinical events**.

## 8. Alert policy (one function, used by evaluation and replay)

- **Scores:**
  - the reading at t, which is the same as the persistence forecast;
  - the trend, physiology, arm B and hybrid forecasts of glucose at +30 minutes;
  - the hybrid classifier's probability of a low starting within 60 minutes.
- **Firing condition at each origin, in time order:**
  - the score passes its threshold: forecast ≤ threshold, or probability ≥ threshold;
  - the reading at t is at least 70 (no alerts during an ongoing low);
  - at least 60 minutes have passed since the last **emitted** alert. Suppressed positives do not reset the cooldown.

  Persistence requirement: 1 origin. Re-arming: none beyond the cooldown. These are software policy choices, not clinical standards.

## 9. Matching alerts to events

- **Warning window.** An alert warns of an event if 0 < event start − alert time ≤ 60 minutes. Alerts after an event has started are never credited.
- **Eligible events.** An event counts for sensitivity only if its whole 60-minute warning window lies inside the observed test interval. The same events are used for every method.
- **One-to-one matching.** Events are taken in time order. Each takes the earliest unmatched alert in its window, which gives that event's lead time.
- **Classifying alerts:**

  | Class | Meaning |
  | --- | --- |
  | Matched | Paired one-to-one with an event |
  | Redundant | Followed by an event within 60 minutes, but not the one matched to that event |
  | False | No event within 60 minutes, with the 60 minutes after the alert fully observed (no gap over 20 minutes, not past the end of the record) |
  | Censored | Follow-up not fully observed; excluded from false-alert counts |

- **Reported:**
  - eligible events and contributing patients;
  - sensitivity;
  - alert precision: (matched + redundant) / (matched + redundant + false);
  - emitted alerts, and false alerts per monitored patient-day. Monitored time is the number of origins × 15 minutes.
  - median and interquartile lead time;
  - 95% intervals from patient-level bootstrap. Resamples with no eligible event leave sensitivity undefined; they are counted and reported, not set to zero.

## 10. Choosing alert thresholds

- **Primary budget:** 0.5 emitted alerts per monitored patient-day. **Secondary budget:** 1.0.
- **Selection.** For each outer fold, every method takes the most permissive threshold whose emitted-alert rate **on the outer-training patients** does not exceed the budget.
  - For learned scores, the training-patient scores are out-of-fold predictions from the inner folds.
  - Selection uses alert rates only, never outcomes.
- **Scoring.** The chosen threshold is applied unchanged to the outer-test patients.
- **Descriptive comparison.** All glucose-type scores are also shown at the fixed thresholds 85 and 90 mg/dL.
- **Classifier.** AUROC, AUPRC and the prevalence of a low within 60 minutes are reported.

## 11. Sensor-off estimation (experimental)

- **Eligible periods.** At least 3 fingersticks in T, and T lasting at least 1.5 days.
- **Starting point.** From the end of C, the core starts from the **last reading inside C**. It sees only:
  - logged doses and meals;
  - fingersticks in T, which re-anchor the path at their times.
- **Hidden CGM.** CGM in T is used only for scoring, never as a state or a feature.
- **Comparators:**
  - the last fingerstick carried forward (before the first fingerstick in T, the last reading in C);
  - the person's typical day: hourly means over C;
  - the core alone;
  - a LightGBM layer combining these, trained on outer-training patients.
- **Reported:**
  - RMSE, MARD and Clarke A+B;
  - error when the reference is below 70, and the share of those readings estimated below 80;
  - error by hours since the last anchor;
  - coverage: days scored per period.
- **Scope.** Coverage is at most about 8 days per period. Nothing here supports claims about weeks of sensor-off use or about acute low alerts without a CGM.

## 12. Reproducibility

- **Run manifest.** `results/manifest.json` records:
  - input file hashes and the protocol hash;
  - configuration, the seed and library versions;
  - exclusions and elapsed time.
- **Saved outputs:**
  - fold assignment, calibration boundaries and offsets;
  - all predictions with origins and target times;
  - emitted alerts.
- **Models.** Fitted per-fold models are saved in `results/models/`. Each model excludes its fold's patients and is used for the live replay.
- **Where every number comes from.** All published numbers are generated from `results/evaluation.json` by code. None is hand-entered.
