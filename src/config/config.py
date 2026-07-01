from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

RunMode = Literal["prototype", "train", "hyperot"]
Frequency = Literal["Daily", "Hourly", "Quarterly", "Monthly", "Weekly"]


class DatasetConfig(BaseModel):
    """Configurações do dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_frequency: Frequency = Field(default="Daily", description="Frequência do dataset")
    impute: bool = Field(default=False, description="Se deve fazer imputação de dados faltantes")


class CentralConfig(BaseModel):
    """Config tipada para o projeto usando pydantic."""

    # frozen=True impede modificações pós-criação, igual ao dataclass(frozen=True)
    model_config = ConfigDict(frozen=True)

    run_mode: RunMode = Field(default="prototype", description="Modo de execução")
    dataset: DatasetConfig = Field(default_factory=DatasetConfig, description="Configurações do dataset")


def load_config(path: Optional[Path] = None) -> CentralConfig:
    """Carrega `config.yaml` do projeto e retorna uma instância tipada `Config`.

    Se `path` não for fornecido, procura `config.yaml` na raiz do projeto.
    """
    if path is None:
        project_root = Path(__file__).resolve().parents[2]
        path = project_root / "config.yaml"

    assert path.exists()

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # O Pydantic recebe o dicionário bruto e faz a validação/conversão de tudo
    return CentralConfig.model_validate(raw)


# Instância carregada ao importar o módulo
try:
    ProjectSettings: CentralConfig = load_config()
    logging.info(ProjectSettings)
except ValidationError as e:
    # O Pydantic lança ValidationError com detalhes amigáveis do que falhou
    raise RuntimeError(f"Falha na validação das configurações:\n{e}") from e
except Exception as e:
    raise RuntimeError(f"Falha ao carregar configuração: {e}") from e
