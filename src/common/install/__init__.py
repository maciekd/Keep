"""The part of the install that runs once there is a Python: `install.sh` hands over with
`python -m common.install` when the venv and its packages are in place.

Bash does what has to happen before that (the machine's checks, the system packages, Python, the
checkout, the venv) and what comes after (the launcher, the summary, the start). What talks to a
person and to an API is here. The package imports nothing of `assistant`, `integrations` or
`interfaces`, and nothing that reads config.yaml — the file does not exist yet, and `ConfigManager`
would write its skeleton."""
