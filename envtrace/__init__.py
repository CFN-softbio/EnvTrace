"""
EnvTrace: Execution-trace alignment and semantic evaluation for code that interacts with environments.

This package provides a domain-agnostic core for:
- Representing execution events and traces (channels/signals over time)
- Aligning traces (e.g., via token-based sequence matchers)
- Computing discrete and continuous metrics (e.g., match rate, timing, profile fidelity)
- Aggregating metric outputs into a full semantic score and a strict binary decision

The core is designed to be lightweight and extensible via plugins/adapters for specific domains
(e.g., EPICS/Bluesky in synchrotron beamlines), while remaining useful in any cyber-physical
or event-driven system.

Project homepage (planned): https://github.com/<org>/envtrace
"""

__all__ = [
    "__version__",
]

__version__ = "0.0.1"
