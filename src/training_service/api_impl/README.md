# Generated route implementations

Keep handwritten implementations in this package, outside the regenerated
tree. Handlers subclass the generated API base interfaces and are imported
by the handwritten application to register those subclasses.

`training_jobs.py` owns submission, listing, details and deletion through the
`JobsBackend` interface in `training_service/jobs.py`. `observability.py` owns
the logs route; `training_service/logs.py` formats snapshots and SSE streams
through the backend's `LogClient` interface. Neither API handler interprets
Ray resources or chooses a Ray SDK client.

Each generated API group selects exactly one registered handler. Missing
handlers and unimplemented base methods return 501; duplicate registrations
fail closed. No per-operation dispatcher or compatibility registry is needed.
Algorithm discovery is implemented. Queue discovery returns 501 until implemented.

Generated routes authenticate and authorize before invoking handlers. Job
resource authorization uses the configured backend's API group and resource;
queue authorization retains its own Kueue resource scope. The
`x-submission-context` and `x-request-context` extensions preserve request
context, submission's 202 status and its Location header.
