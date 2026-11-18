---
name: commit
description: Commit and push all current changes with an auto-generated commit messages
disable-model-invocation: false
allowed-tools: Bash(git *)
effort: medium
argument-hint: "[why this change was made]"
---

Commit changes

Optional argument — the user's own explanation of why the change was made: $ARGUMENTS

If an argument is given, treat it as the primary source for the WHY in the commit body. It is
usually typed quickly and informally — do not paste it verbatim. Clean it up: fix typos and
grammar, expand shorthand, fold it into well-formed sentences, and combine it with what the diff
and the conversation tell you. Preserve its intent and any specifics (numbers, names, constraints)
exactly. If the argument is empty, derive the WHY from the diff and the conversation as usual.

1. First, run 'git diff' to see all changes (both staged and unstaged)
2. Analyze the diff to understand what changed
3. Analyze past discussion with user (and the argument above, if given) to understand the reason behind the commit better
4. Write an excellent commit message
   - First line (the "subject") is short and concise. Start with type of commit (feat, fix, refactor, docs, cleanup or test) and a colon.
   - Add a description (the "body"/"description") that details WHY the user wanted to do this and other details that can be of interest (act as an experienced developer that knows what's of value in a commit message)
5. Stage all changes with 'git add -A'
6. Commit with the commit message
7. Push the changes

Split into several commits if several unrelated changes are made.

Sometimes debug changes are marked with "NOCOMMIT" in the code. If you find these: temporarily remove them, commit and then put the debug changes back in.

Note that prompt.md and testsuite_prompt.md are system prompts for AI, not docs.
