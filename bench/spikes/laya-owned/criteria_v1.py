"""V1 criteria, derived from the Athena tool's own published description.

Source, verbatim from the athena tool description available to every seat:
  "Athena classifies the request and runs the right shape: design review, audit,
   diagnosis, evaluation, explanation, or an optimization campaign (a time-boxed
   experiment loop: Athena plans the experiment, returns the plan to you for
   approval, ... measures panel-proposed changes, and reports the leaderboard)."

That is a real source describing the six shapes, not wording invented from the
label names, and it was fixed before any V1 accuracy was seen. It is weaker than
the classify prompt itself, which does not exist in this repo; ALF has been asked
for that text and it would run as V2.
"""
CRITERIA_V1 = {
    "AUDIT": "audit an existing diff, change or implementation for defects and regressions",
    "EVALUATE": "evaluate or compare options, or judge whether something is good enough",
    "PLAN": "review a proposed design before it is built, and settle the approach",
    "DIAGNOSE": "diagnose why something is failing or behaving unexpectedly",
    "EXPLAIN": "explain how an existing system works",
    "SPEC": "run a time-boxed optimization campaign that measures proposed changes against a metric",
}
