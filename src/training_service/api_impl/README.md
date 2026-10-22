# Generated route implementations

Keep handwritten implementations in this package, outside the regenerated
tree. Future handlers subclass the generated API base interfaces and are
imported by the handwritten application to register those subclasses.

`training_jobs.py` implements listing, details, and terminal RayJob deletion;
`observability.py` implements driver logs through the Ray SDK. Generated
routes authenticate and authorize before dispatch. Request context is passed
explicitly for operations marked `x-request-context` in the contract. Other
operations remain fail-closed pending their own implementation tasks.
