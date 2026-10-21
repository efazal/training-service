# Generated route implementations

Keep handwritten implementations in this package, outside the regenerated
tree. Future handlers subclass the generated API base interfaces and are
imported by the handwritten application to register those subclasses.

The Training Jobs handler implements POST submission through the caller-scoped
CodeFlare adapter in `training_service/submission.py`. List, get, and delete
remain fail-closed with `501` until their own work is merged. Authentication
and project authorization run in generated routes before this handler.
