"""Re-export shim — causal_providers was promoted to ``cognitive_os``.

See the companion ``causal_attribution`` shim. The provider + sandbox binding
layer is now ``cognitive_os/causal_providers.py``; this preserves the historical
top-level import for the ARC harness and the engine test suite.
"""
from cognitive_os.causal_providers import *             # noqa: F401,F403
from cognitive_os.causal_providers import (              # explicit (used names)
    Observation, Sandbox, CopySandbox, UndoSandbox, SandboxProvider,
    TIER_BY_KIND, to_relation,
)
