"""
presentation_codes.py
=====================
Exact translation of value_getPresentationCodes.m (H Atilgan & AC Kwan, 191210).

Maps NBS Presentation event codes for each task phase.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StimCodes:
    GO: int = 21


@dataclass
class RespCodes:
    MANUAL: int = 1   # experimenter manual reward
    LEFT: int = 2
    RIGHT: int = 3


@dataclass
class OutcomeCodes:
    REWARDLEFT: int = 5
    REWARDRIGHT: int = 6
    REWARDMANUAL: int = 7   # manual reward = treated as miss
    MISS: int = 8
    NOREWARDLEFT: int = 75
    NOREWARDRIGHT: int = 76


@dataclass
class RuleCodes2:
    """Two reward probability sets (Phase 3, 8, 21, 22, 31, 32)."""
    L70R10: int = 41
    L10R70: int = 42


@dataclass
class RuleCodes6:
    """Six reward probability sets (Phase 6)."""
    L70R30: int = 41
    L70R10: int = 42
    L30R10: int = 43
    L30R70: int = 44
    L10R70: int = 45
    L10R30: int = 46


@dataclass
class EventCodes:
    WAITCUE: int = 90        # inter-trial no-lick period
    SESSIONSTART: Optional[int] = None
    SESSIONEND: Optional[int] = None
    LASERON: Optional[int] = None
    LASEROFF: Optional[int] = None


def get_presentation_codes(phase: int):
    """
    Exact translation of value_getPresentationCodes.m.

    Parameters
    ----------
    phase : int
        Task phase code (3, 6, 8, 21, 22, 31, 32).

    Returns
    -------
    stim, resp, outcome, rule, event
    """
    stim = StimCodes()
    resp = RespCodes()
    outcome = OutcomeCodes()
    event = EventCodes()

    if phase in (3, 8, 1, 2):
        # Reversal Version 9/30/18
        rule = RuleCodes2(L70R10=41, L10R70=42)

    elif phase in (21, 22):
        # Reversal + OptoLaser for Cg1/M2
        rule = RuleCodes2(L70R10=41, L10R70=42)
        event.SESSIONSTART = 61
        event.SESSIONEND = 60
        event.LASERON = 63
        event.LASEROFF = 62

    elif phase == 6:
        # Dynamic foraging with 6 reward probability sets
        rule = RuleCodes6()

    elif phase in (31, 32):
        # Reversal + NE recording
        rule = RuleCodes2(L70R10=41, L10R70=42)

    else:
        raise ValueError(f"Unknown phase code: {phase}. "
                         f"Expected one of: 3, 6, 8, 21, 22, 31, 32.")

    return stim, resp, outcome, rule, event


# Convenience: reward probability lookup per rule index
REWARD_PROBS = {
    2: {1: (0.7, 0.1), 2: (0.1, 0.7)},   # nRules=2: (left_prob, right_prob)
    6: {1: (0.7, 0.3), 2: (0.7, 0.1), 3: (0.3, 0.1),
        4: (0.3, 0.7), 5: (0.1, 0.7), 6: (0.1, 0.3)},
}

RULE_LABELS = {
    2: {1: "0.7:0.1", 2: "0.1:0.7"},
    6: {1: "0.7:0.3", 2: "0.7:0.1", 3: "0.3:0.1",
        4: "0.3:0.7", 5: "0.1:0.7", 6: "0.1:0.3"},
}
