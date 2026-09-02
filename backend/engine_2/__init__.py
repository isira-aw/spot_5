"""engine_2 — the quantitative brain: a CNN-BiLSTM-MHA forecaster and the PPO
policy trained against it.

This package is a MODEL FACTORY and an inference loop. It fetches public market
data, trains, gates, versions and promotes model bundles, and scores bars. It
contains no order placement, no execution and no withdrawal path — trading lives
in `backend/execution`, driven by the Agent, and engine_2 only ever hands it an
opinion. See README.md.
"""
