"""Re-export shim — causal_attribution was promoted to ``cognitive_os``.

The module is now domain-general engine substrate (a second consumer — robotics
— confirmed the interface), so it lives at
``cognitive_os/causal_attribution.py``.  This shim preserves the historical
top-level import (`import causal_attribution` / `from causal_attribution import
...`) used by the ARC harness and the engine test suite, forwarding every name
to the engine module so there is a single source of truth and one class
identity across both import paths.

Migrate call sites to ``from cognitive_os.causal_attribution import ...`` at
leisure; this shim can be removed once none remain.  See
``usecases/robotics/NOTES_causal_loop_promotion.md``.
"""
from cognitive_os.causal_attribution import *           # noqa: F401,F403
from cognitive_os.causal_attribution import (            # explicit (private/used)
    Relation, Anomaly, Suspect, CausalClaim, Provider,
    detect_anomaly, generate_suspects, narrow_by_contrast,
    find_negating_actions, confirm_by_counterfactual, attribute,
)
