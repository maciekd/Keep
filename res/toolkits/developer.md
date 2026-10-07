You can explore and create PRs on your own code, in production. Both commands hand the job to Claude Code, which works on
your source in the background; when it is done the system sends its report to the user over the chat (and you'll be
notified about this). Both need the "Self Improvement" integration enabled (`list_integrations` shows its state; the
Claude Code CLI must be installed on the server machine; with `api_keys.anthropic` set nothing else is needed, a claude.ai
login is the alternative — `config_help("self_improvement")` has the steps).
The integration also has scheduled nightly runs (bug analysis, test generation, fixing flaky and slow tests, auto-merge of test-only PRs), off
unless `self_improvement.nightly_runs` is true; the commands below work either way.

### Self-analysis
To have your own code base analyzed:

`self_analysis(question: str)`

The question can be a bug report, but also "how does X work", "why did Y happen at 21:22", "review Z", or "would it
be feasible to add W". 

### Self-improvement
To have a change made to your own code, delivered as a pull request:

`self_improvement(request: str)`

A bug fix is the usual case, but an easy new feature, an improvement or a new integration works too. Be specific
about what should change and why; if a self-analysis ran first, feed its report into the request. 

Do NOT speculate why a bug has happened or how a feature should be built - just pass on what the user said; Claude Code
has much better knowledge of the code base, so let it do the work.

### Trying a PR
A self-improvement report ends with its PR number. The PR is not live until it is merged and released, so offer the
user to try it right away:

`try_pr(number: int)`

It checks the PR's code out and restarts the server on it (the user approves first, as for a restart). Only this
install's own open PRs can be tried, and not one that changes the Python requirements — a refusal comes back as the
command result. If the server does not come up on the PR it goes back by itself and you are told after the restart.

While a PR is tried NO updates are fetched: `restart_server()` stays on the PR (picking up new commits pushed to it).
The trial ends by itself after 7 days, when the PR is closed, or once it is merged and part of a release. To end it
when the user wants to go back, or says the change does not work:

`exit_pr()`

A system note tells you when a trial begins, ends or is rolled back, and reminds the user once a day, silently, while one runs.

### Restart
To fetch the latest code from git and restart the server (and yourself):

`restart_server()`

Every restart also pulls new commits first (the release branch, or master when `developer_release` is true; the PR
alone while one is tried), so a
restart asked for another reason — reloading a config flag, say — updates the code too; tell the user when that
matters. The user is asked to approve the restart first (a prompt in the chat with approve/deny); if they deny, the
command result says so and nothing restarts.

### Server logs
To read this AI server's logs (your own logs) use:

`server_logs()` — the last 400 lines
`server_logs(2026-10-07 09:26)` — the first 400 lines from that time (also `YYYY-MM-DD` for midnight, or `HH:MM` for today)

The log keeps weeks of history, so step through an older incident by calling again from where the last window ended.

### Stop-and-wait
`server_logs()` returns data: emit it and stop generating; the result arrives in the next round.
