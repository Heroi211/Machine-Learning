"""Template Method para pipelines reprodutíveis (DVC, CLI, orquestradores)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.ml.run_context import ModelCandidate, RunContext, RunResult


class PipelineRunner(ABC):
    runner_id: str

    @abstractmethod
    def preprocess(self, ctx: RunContext) -> Any: ...

    @abstractmethod
    def feature_eng(self, data: Any, ctx: RunContext) -> Any: ...

    @abstractmethod
    def train(self, data: Any, ctx: RunContext) -> list[ModelCandidate]: ...

    @abstractmethod
    def evaluate(self, candidates: list[ModelCandidate], ctx: RunContext) -> ModelCandidate: ...

    def run(self, ctx: RunContext) -> RunResult:
        data = self.preprocess(ctx)
        data = self.feature_eng(data, ctx)
        candidates = self.train(data, ctx)
        champion = self.evaluate(candidates, ctx)
        mlflow_run_id = self.log_mlflow(champion, ctx)
        return RunResult(
            manifest=champion.manifest,
            metrics=champion.metrics,
            mlflow_run_id=mlflow_run_id,
            champion_name=champion.name,
        )

    def run_stage(self, stage: str, ctx: RunContext) -> Any:
        if stage == "preprocess":
            return self.preprocess(ctx)
        if stage == "feature_eng":
            data = self.preprocess(ctx)
            return self.feature_eng(data, ctx)
        if stage == "train":
            data = self.feature_eng(self.preprocess(ctx), ctx)
            return self.train(data, ctx)
        if stage == "evaluate":
            data = self.feature_eng(self.preprocess(ctx), ctx)
            candidates = self.train(data, ctx)
            return self.evaluate(candidates, ctx)
        if stage == "all":
            return self.run(ctx)
        raise ValueError(f"Stage desconhecido: {stage!r}")

    def log_mlflow(self, champion: ModelCandidate, ctx: RunContext) -> str | None:
        return None


RUNNER_REGISTRY: dict[str, type[PipelineRunner]] = {}


def register_runner(runner_cls: type[PipelineRunner]) -> type[PipelineRunner]:
    RUNNER_REGISTRY[runner_cls.runner_id] = runner_cls
    return runner_cls


def get_runner(runner_id: str) -> PipelineRunner:
    runner_cls = RUNNER_REGISTRY.get(runner_id)
    if runner_cls is None:
        known = ", ".join(sorted(RUNNER_REGISTRY)) or "(vazio)"
        raise KeyError(f"PipelineRunner {runner_id!r} não registrado. Disponíveis: {known}")
    return runner_cls()
