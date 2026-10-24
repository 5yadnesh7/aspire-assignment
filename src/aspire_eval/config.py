import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ProviderConfig(BaseModel):
    name: str
    base_url: str
    api_key_env: str
    pricing_source: str
    pricing_retrieved_at: str
    currency: str
    price_unit: str
    native_schema_constrained_output: bool
    structured_output_strategy: str


class Pricing(BaseModel):
    input: float
    cached_input: float
    output: float


class ModelConfig(BaseModel):
    model: str
    catalog_digest: str
    capabilities: list[str]
    pricing_usd_per_1m_tokens: Pricing


class ScoreWeights(BaseModel):
    coverage: float = Field(gt=0, lt=1)
    correctness: float = Field(gt=0, lt=1)
    bug_detection: float = Field(gt=0, lt=1)

    def validate_total(self) -> None:
        total = self.coverage + self.correctness + self.bug_detection
        if abs(total - 1) > 1e-9:
            raise ValueError(f"Score weights must sum to 1, got {total}")


class PipelineSettings(BaseModel):
    evaluation_runs: int = Field(ge=1)
    temperature: float = Field(ge=0, le=2)
    max_validation_retries: int = Field(ge=0)
    preserve_every_raw_response: bool
    quality_target: float = Field(ge=0, le=100)
    score_weights: ScoreWeights


class UsageFields(BaseModel):
    input_tokens: str
    cached_input_tokens: str
    output_tokens: str
    latency_nanoseconds: str


class CostTracking(BaseModel):
    formula: str
    uncached_input_tokens_formula: str
    record_by: list[str]
    retry_costs_are_included: bool
    max_estimated_usd: float = Field(gt=0)
    missing_cached_token_policy: str


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str
    provider: ProviderConfig
    pipeline: PipelineSettings
    models: dict[str, ModelConfig]
    usage_fields: UsageFields
    cost_tracking: CostTracking


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config: Path
    outputs: Path


def project_root() -> Path:
    here = Path.cwd()
    for candidate in (here, *here.parents):
        if (candidate / "config" / "models.json").exists():
            return candidate
    raise FileNotFoundError("config/models.json was not found from the current directory")


def load_project_env(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(root: Path | None = None) -> tuple[ProjectPaths, AppConfig]:
    project = root or project_root()
    load_project_env(project)
    config_path = project / "config" / "models.json"
    config = AppConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    config.pipeline.score_weights.validate_total()
    paths = ProjectPaths(project, config_path, project / "outputs")
    return paths, config
