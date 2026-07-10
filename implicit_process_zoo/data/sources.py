"""Auditable source metadata for datasets downloaded by this project."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataSource:
    name: str
    url: str
    sha256: str
    filename: str
    members: tuple[str, ...] = ()
    doi: str | None = None
    license: str | None = None
    attribution: str | None = None


DATA_SOURCES = {
    "eld": DataSource(
        name="ElectricityLoadDiagrams20112014",
        url="https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
        sha256="F6C4D0E0DF12ECDB9EA008DD6EEF3518ADB52C559D04A9BAC2E1B81DCFC8D4E1",
        filename="electricityloaddiagrams20112014.zip",
        members=("LD2011_2014.txt",),
        doi="10.24432/C58C86",
        license="CC BY 4.0",
        attribution="Artur Trindade",
    ),
    "boston": DataSource(
        "Boston Housing",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
        "BAADF72995725D76EFE787B664E1F083388C79BA21EF9A7990D87F774184735A",
        "housing.data",
    ),
    "energy": DataSource(
        "Energy Efficiency",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
        "0089FCFFC1415E41E2FF63730CCA5280EFE54FC43D722DFDE1B0AAA808E35DC4",
        "ENB2012_data.xlsx",
    ),
    "concrete": DataSource(
        "Concrete Compressive Strength",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
        "710076C66B9CA3F8050E7942F3DCBDBE04013534DAEB0077FFD3079A52D8E0C4",
        "Concrete_Data.xls",
    ),
    "naval": DataSource(
        "Condition Based Maintenance of Naval Propulsion Plants",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
        "91A3815DA80B5AB7E2D5B82AC82F1C2CBF89182C7A65BCDF240DB1E014423CB9",
        "UCI_CBM_Dataset.zip",
        ("UCI CBM Dataset/data.txt",),
    ),
    "power": DataSource(
        "Combined Cycle Power Plant",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
        "CC7B2A4977C0A44E8221C91D9A7E5746B3C68186CFF7E5C61C70AF6432B98C7A",
        "CCPP.zip",
        ("CCPP/Folds5x2_pp.xlsx",),
    ),
    "protein": DataSource(
        "Physicochemical Properties of Protein Tertiary Structure",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
        "4277CFCB4E91A181746CBC654F001B57951C9E6A80F4F795FDB5C807E0848F40",
        "CASP.csv",
    ),
    "kin8nm": DataSource(
        "Kin8nm",
        "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff",
        "7B9BF0301AC936D88122557A151E1BA8F1EBC278FCF46D9F3C6D462DEBDBC8AD",
        "kin8nm.csv",
    ),
    "yacht": DataSource(
        "Yacht Hydrodynamics",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
        "00DFECC0FC01DDD4C90B558A3AC11B246DF8EBCFEA130724223475A9A67F0EA1",
        "yacht_hydrodynamics.data",
    ),
    "winered": DataSource(
        "Wine Quality Red",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "4A402CF041B025D4566D954C3B9BA8635A3A8A01E039005D97D6A710278CF05E",
        "winequality-red.csv",
    ),
}


def get_data_source(name: str) -> DataSource:
    try:
        return DATA_SOURCES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown data source {name!r}; choose from {sorted(DATA_SOURCES)}") from exc
