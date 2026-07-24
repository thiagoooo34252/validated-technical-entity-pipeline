"""Validated domain schema for technical entity extraction."""

from enum import Enum
from typing import Annotated, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Technology = Annotated[str, StringConstraints(min_length=1, max_length=100)]
TechnicalSummary = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=10, max_length=1000)
]


class NivelDeCriticidad(str, Enum):  # noqa: UP042 - the contract explicitly requires str, Enum
    """Supported criticality levels, serialized with the required Spanish values."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """A validated collection of technologies, criticality, and technical context."""

    model_config = ConfigDict(extra="forbid")

    tecnologias: list[Technology] = Field(
        min_length=1,
        max_length=50,
        description=(
            "Distinct technical products, platforms, languages, or frameworks found in the text."
        ),
    )
    nivel_de_criticidad: NivelDeCriticidad = Field(
        description="Technical impact level using one of the required Spanish enum values.",
    )
    resumen_tecnico: TechnicalSummary = Field(
        description="A concise, non-empty technical summary grounded only in the source text.",
    )

    @field_validator("tecnologias", mode="before")
    @classmethod
    def normalize_technologies(cls, value: object) -> list[str]:
        """Normalize whitespace and remove case-insensitive duplicates in source order."""
        if not isinstance(value, list):
            raise ValueError("tecnologias must be a list")

        normalized: list[str] = []
        seen: set[str] = set()
        for item in cast(list[object], value):
            if not isinstance(item, str):
                raise ValueError("each technology must be a string")

            technology = " ".join(item.split())
            if not technology:
                raise ValueError("technology values must not be blank")

            identity = technology.casefold()
            if identity not in seen:
                seen.add(identity)
                normalized.append(technology)

        return normalized
