"""Behavior tests for the extraction domain schema."""

import pytest
from pydantic import ValidationError

from schemas import ExtraccionTecnica, NivelDeCriticidad


def test_schema_exposes_exact_required_fields() -> None:
    assert list(ExtraccionTecnica.model_fields) == [
        "tecnologias",
        "nivel_de_criticidad",
        "resumen_tecnico",
    ]


def test_schema_accepts_valid_extraction() -> None:
    extraction = ExtraccionTecnica(
        tecnologias=["Python", "PostgreSQL"],
        nivel_de_criticidad="alta",
        resumen_tecnico="The service depends on Python and PostgreSQL for ingestion.",
    )

    assert extraction.tecnologias == ["Python", "PostgreSQL"]
    assert extraction.nivel_de_criticidad is NivelDeCriticidad.ALTA
    assert '"nivel_de_criticidad":"alta"' in extraction.model_dump_json()


@pytest.mark.parametrize("level", ["baja", "media", "alta"])
def test_schema_accepts_every_criticality_value(level: str) -> None:
    extraction = ExtraccionTecnica(
        tecnologias=["Linux"],
        nivel_de_criticidad=level,
        resumen_tecnico="Linux is present in the described runtime environment.",
    )

    assert extraction.nivel_de_criticidad.value == level


def test_schema_rejects_empty_technology_list() -> None:
    with pytest.raises(ValidationError):
        ExtraccionTecnica(
            tecnologias=[],
            nivel_de_criticidad="baja",
            resumen_tecnico="No usable technology was extracted from this source.",
        )


def test_schema_accepts_maximum_technology_list_length() -> None:
    technologies = [f"Technology-{index:02d}" for index in range(50)]

    extraction = ExtraccionTecnica(
        tecnologias=technologies,
        nivel_de_criticidad="media",
        resumen_tecnico="The extraction contains the maximum supported technology count.",
    )

    assert extraction.tecnologias == technologies


def test_schema_rejects_technology_list_above_maximum_length() -> None:
    technologies = [f"Technology-{index:02d}" for index in range(51)]

    with pytest.raises(ValidationError) as error:
        ExtraccionTecnica(
            tecnologias=technologies,
            nivel_de_criticidad="media",
            resumen_tecnico="The extraction exceeds the supported technology count.",
        )

    assert error.value.errors()[0]["type"] == "too_long"


def test_schema_rejects_non_list_technologies() -> None:
    with pytest.raises(ValidationError, match="tecnologias must be a list"):
        ExtraccionTecnica(
            tecnologias="Python",  # type: ignore[arg-type]
            nivel_de_criticidad="baja",
            resumen_tecnico="Python appears as an invalid scalar technology collection.",
        )


def test_schema_normalizes_and_deduplicates_technologies_in_order() -> None:
    extraction = ExtraccionTecnica(
        tecnologias=["  Python  ", "python", "PostgreSQL   16", " POSTGRESQL 16 "],
        nivel_de_criticidad="media",
        resumen_tecnico="  Python communicates with the PostgreSQL 16 database.  ",
    )

    assert extraction.tecnologias == ["Python", "PostgreSQL 16"]
    assert extraction.resumen_tecnico == "Python communicates with the PostgreSQL 16 database."


def test_schema_rejects_blank_technology_value() -> None:
    with pytest.raises(ValidationError, match="technology values must not be blank"):
        ExtraccionTecnica(
            tecnologias=["Python", "   "],
            nivel_de_criticidad="media",
            resumen_tecnico="Python is the only valid technology in this extraction.",
        )


def test_schema_rejects_non_string_technology_value() -> None:
    with pytest.raises(ValidationError, match="each technology must be a string"):
        ExtraccionTecnica(
            tecnologias=["Python", 42],  # type: ignore[list-item]
            nivel_de_criticidad="media",
            resumen_tecnico="Python is accompanied by an invalid non-string technology.",
        )


def test_schema_accepts_maximum_individual_technology_length() -> None:
    technology = "T" * 100

    extraction = ExtraccionTecnica(
        tecnologias=[technology],
        nivel_de_criticidad="baja",
        resumen_tecnico="The technology name is exactly at the supported character limit.",
    )

    assert extraction.tecnologias == [technology]


def test_schema_rejects_individual_technology_above_maximum_length() -> None:
    with pytest.raises(ValidationError) as error:
        ExtraccionTecnica(
            tecnologias=["T" * 101],
            nivel_de_criticidad="baja",
            resumen_tecnico="The technology name exceeds the supported character limit.",
        )

    assert error.value.errors()[0]["type"] == "string_too_long"


def test_schema_rejects_blank_summary() -> None:
    with pytest.raises(ValidationError):
        ExtraccionTecnica(
            tecnologias=["Python"],
            nivel_de_criticidad="media",
            resumen_tecnico="   ",
        )


def test_schema_accepts_maximum_technical_summary_length() -> None:
    summary = "S" * 1000

    extraction = ExtraccionTecnica(
        tecnologias=["Python"],
        nivel_de_criticidad="media",
        resumen_tecnico=summary,
    )

    assert extraction.resumen_tecnico == summary


def test_schema_rejects_technical_summary_above_maximum_length() -> None:
    with pytest.raises(ValidationError) as error:
        ExtraccionTecnica(
            tecnologias=["Python"],
            nivel_de_criticidad="media",
            resumen_tecnico="S" * 1001,
        )

    assert error.value.errors()[0]["type"] == "string_too_long"
