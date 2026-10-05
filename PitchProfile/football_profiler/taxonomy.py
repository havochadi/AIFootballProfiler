"""Versioned proposal vocabulary; catalogue membership is not model validation."""
import json
import math
from pathlib import Path

CATALOGUE = json.loads((Path(__file__).resolve().parents[1] / 'annotation/archetypes.v2.json').read_text(encoding='utf-8'))
VERSION = CATALOGUE['version']
ROLES = {role['id']: role for role in CATALOGUE['roles']}
GROUPS = {group['id']: group['name'] for group in CATALOGUE['position_groups']}
LABELS = tuple(ROLES)
# Independent per-role percentage ratings within this many points of each other count as
# reviewer agreement; wider spreads need adjudication. See football_profiler/cases.py:consensus.
AGREEMENT_TOLERANCE = 20
# Consensus value at or above this counts as a secondary trait for primary/secondary display.
SECONDARY_FLOOR = 30


def compatible(group):
    if group not in GROUPS:
        raise ValueError('Choose a reviewed position group from the catalogue')
    return tuple(key for key, role in ROLES.items() if group in role['position_groups'])


def validate_labels(group, labels):
    allowed = compatible(group)
    if set(labels) != set(allowed):
        raise ValueError('Rate each role compatible with this case position; leave unsupported roles unknown')
    for value in labels.values():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError('Each role must be a percentage from 0 to 100, or null for insufficient evidence')
