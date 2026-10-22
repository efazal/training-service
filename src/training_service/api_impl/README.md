# Generated route implementations

Keep handwritten implementations in this package, outside the regenerated
tree. Handlers subclass the generated API base interfaces and are imported
by the handwritten application to register those subclasses.

`training_jobs.py` implements POST through the caller-scoped CodeFlare adapter
in `training_service/job_submission.py`. `job_operations.py` implements
listing, details, and terminal RayJob deletion; `observability.py` implements
driver logs through the Ray SDK. Generated routes authenticate and authorize
before dispatch. `x-submission-context` passes POST request context and
preserves the 202/Location response; `x-request-context` passes request context
for job operations. Algorithm/queue discovery and estimation remain
fail-closed pending their own implementation tasks.

Handlers declare `implemented_operations` so generated routes select each
operation's owner independently of import order. The separate submission
handler coexists with reads without its placeholder methods shadowing them.
See `dispatch.py` for the legacy-handler compatibility path.
