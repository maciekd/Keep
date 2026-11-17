#!/bin/bash
# install.sh's test double for the checkout's start.sh, symlinked into it: says in
# KEEP_TEST_STARTED that it ran, and whether it was given a terminal.
if [ -t 0 ]; then echo terminal > "$KEEP_TEST_STARTED"; else echo none > "$KEEP_TEST_STARTED"; fi
