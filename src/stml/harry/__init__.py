"""Harry's contribution to the stml metamodel.

This subpackage is the entry point for Harry's synthesis layer: a signal-direction
audit, a canonical triple-barrier labeller with next-day execution, a novel
feature pack (signal-trajectory, conditional-risk, information-theoretic,
microstructure, cross-asset, wavelet, concept-drift, optional TDA), an
independent end-to-end pipeline producing predictions for the released window,
and an optional strategy track.

Everything in this subpackage is leakage-safe (truncation-invariant) and
deterministic (default `random_state=42`). It depends on `stml.io` and
`stml.na_checks` from the shared foundation and treats every other branch's
modules as read-only references.
"""

__all__: list[str] = []
