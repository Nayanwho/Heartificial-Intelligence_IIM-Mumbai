# Superseded results (version 1, 19 October 2026)

These files are the original evaluation, kept for the record. **Do not quote them.**
An audit on 23 October found that this evaluation let later data into earlier inputs:

- the sensor offset used the whole period's fingersticks;
- the offset prior included test patients;
- missing meals used a whole-period median;
- the calibration length depended on the period length;
- sensor-off mode started from a hidden CGM reading.

Its alarm scoring also credited alerts and counted false alarms under different rules.

The current results are in `results/`, produced by `scripts/evaluate.py` under `docs/evaluation_protocol.md`.
The version-1 code is in `scripts/legacy/evaluate_v1.py`.
