# GEPA Optimizer Environment

This package wires the core `verifiers` training loop to the GEPA prompt
optimizer. It exposes a `load_environment` factory that returns an instance of
`vf.GepaEnvironment`, which dynamically selects and refines system prompts
mid-training.

Refer to `environments/gepa_optimizer/gepa_env.py` for configuration entry
points. Default datasets are not bundled; provide your own `datasets.Dataset`
objects when calling `load_environment`.
