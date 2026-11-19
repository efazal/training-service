"""A source-serializable callback; all dependencies are imported inside methods."""

from training_hub.callbacks import TrainingHubCallback


class MLflowMetricsCallback(TrainingHubCallback):
    """Forward normalized lifecycle events across in-process and torchrun trainers."""

    run_on_all_ranks = True

    def _event(self, event: str, context: object) -> None:
        from training_service_metrics_runtime import current_recorder

        current_recorder().event(event, context)

    def on_train_begin(self, context: object) -> None:
        self._event("on_train_begin", context)

    def on_step_begin(self, context: object) -> None:
        self._event("on_step_begin", context)

    def on_step_end(self, context: object) -> None:
        self._event("on_step_end", context)

    def on_log(self, context: object) -> None:
        self._event("on_log", context)

    def on_evaluate(self, context: object) -> None:
        self._event("on_evaluate", context)

    def on_save(self, context: object) -> None:
        self._event("on_save", context)

    def on_epoch_begin(self, context: object) -> None:
        self._event("on_epoch_begin", context)

    def on_epoch_end(self, context: object) -> None:
        self._event("on_epoch_end", context)

    def on_train_end(self, context: object) -> None:
        self._event("on_train_end", context)
        # A torchrun subprocess must flush before it exits. The driver owns run status.
        import os
        import socket

        driver = os.getenv("TRAINING_SERVICE_MLFLOW_DRIVER_PROCESS")
        if driver and driver != f"{socket.gethostname()}:{os.getpid()}":
            from training_service_metrics_runtime import current_recorder

            current_recorder().close()
