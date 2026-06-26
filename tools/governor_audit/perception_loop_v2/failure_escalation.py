"""failure_escalation.py -- a game-agnostic backstop against the 'guess again' trap.

When the loop keeps acting with NO progress -- retrying the same action, or varying a detail of the
same approach turn after turn without the world advancing -- it should ESCALATE: stop repeating and
switch strategy.  The single most useful redirection is "the answer is usually DISPLAYED -- MEASURE
the ground truth (a reference/key panel, or a provided solution image) instead of guessing."  This is
the meta-level safety net behind [[feedback_read_dont_guess_displayed_answer]]; it caught nothing in
the tn36-lc2 RIGHT-code episode because there was none.

Pure logic: the driver feeds (action_signature, progressed) each turn -- where `progressed` is True
iff that action advanced the world (score/lc up, OR a genuinely NEW state reached) -- and reads
directive() to surface an escalation note into the next prompt.
"""
from __future__ import annotations

from collections import Counter


class FailureEscalator:
    def __init__(self, repeat_threshold: int = 3, stagnation_turns: int = 6, window: int = 14):
        self.repeat_threshold = repeat_threshold   # same action N times, no progress -> escalate
        self.stagnation_turns = stagnation_turns    # M turns, zero progress -> escalate
        self.window = window
        self.hist: list = []                        # [(sig, progressed)]

    def record(self, action_signature, progressed: bool) -> None:
        self.hist.append((str(action_signature), bool(progressed)))
        if len(self.hist) > self.window:
            self.hist = self.hist[-self.window:]

    def _repeated_failure(self):
        """A specific action retried >= threshold times with NO progress ever (in window)."""
        progressed = {s for s, p in self.hist if p}
        fails = Counter(s for s, p in self.hist if not p)
        for sig, cnt in fails.items():
            if cnt >= self.repeat_threshold and sig not in progressed:
                return sig, cnt
        return None

    def _stagnant(self) -> bool:
        """The last M turns made zero progress (any approach, not just one action)."""
        recent = self.hist[-self.stagnation_turns:]
        return len(recent) >= self.stagnation_turns and not any(p for _, p in recent)

    def directive(self):
        """An escalation note for the prompt, or None.  Repeated-failure takes priority (more
        specific) over stagnation."""
        rf = self._repeated_failure()
        if rf:
            sig, cnt = rf
            return (f"REPEATED-FAILURE ESCALATION: action '{sig}' has been tried {cnt}x with NO "
                    f"score/lc progress. STOP retrying it. The answer is usually DISPLAYED -- MEASURE "
                    f"the ground truth (a reference/key panel showing the active option's code, or a "
                    f"provided solution image) instead of guessing, or take a structurally DIFFERENT "
                    f"action. Do not repeat '{sig}'.")
        if self._stagnant():
            return (f"STAGNATION ESCALATION: {self.stagnation_turns} turns with NO score/lc progress. "
                    f"Stop varying the same approach. RE-READ the scene's own instructions/reference "
                    f"by MEASUREMENT (the game usually DISPLAYS what to do), or switch to a "
                    f"structurally different strategy.")
        return None
