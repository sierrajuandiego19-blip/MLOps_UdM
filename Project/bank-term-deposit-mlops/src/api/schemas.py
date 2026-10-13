"""
Contratos de entrada y salida de la API (Pydantic v2).

La validación es estricta a propósito: los campos categóricos se declaran como
`Literal` con las categorías exactas del dataset y los numéricos tienen cotas.
Una petición mal formada se rechaza con un 422 que dice qué campo falla, en vez
de llegar al modelo y producir una predicción silenciosamente inválida.

`extra="forbid"` cierra el contrato: un campo con el nombre mal escrito es un
error, no algo que se ignore.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Categorías exactas del dataset UCI Bank Marketing.
JobType = Literal[
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]
MaritalType = Literal["divorced", "married", "single", "unknown"]
EducationType = Literal[
    "basic.4y",
    "basic.6y",
    "basic.9y",
    "high.school",
    "illiterate",
    "professional.course",
    "university.degree",
    "unknown",
]
YesNoUnknown = Literal["no", "yes", "unknown"]
ContactType = Literal["cellular", "telephone"]
MonthType = Literal[
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
]
DayOfWeekType = Literal["mon", "tue", "wed", "thu", "fri"]
PoutcomeType = Literal["failure", "nonexistent", "success"]


class ClientFeatures(BaseModel):
    """Datos de un cliente más el contexto de la campaña."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 41,
                "job": "technician",
                "marital": "married",
                "education": "university.degree",
                "default": "no",
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "month": "may",
                "day_of_week": "thu",
                "campaign": 2,
                "pdays": 999,
                "previous": 0,
                "poutcome": "nonexistent",
                "emp.var.rate": -1.8,
                "cons.price.idx": 92.893,
                "cons.conf.idx": -46.2,
                "euribor3m": 1.313,
                "nr.employed": 5099.1,
            }
        },
    )

    # --- Datos del cliente ---
    age: int = Field(..., ge=17, le=100, description="Edad en años")
    job: JobType = Field(..., description="Tipo de trabajo")
    marital: MaritalType = Field(..., description="Estado civil")
    education: EducationType = Field(..., description="Nivel educativo")
    default: YesNoUnknown = Field(..., description="¿Tiene crédito en mora?")
    housing: YesNoUnknown = Field(..., description="¿Tiene crédito hipotecario?")
    loan: YesNoUnknown = Field(..., description="¿Tiene crédito personal?")

    # --- Contexto del contacto ---
    contact: ContactType = Field(..., description="Canal de contacto")
    month: MonthType = Field(..., description="Mes del último contacto")
    day_of_week: DayOfWeekType = Field(..., description="Día de la semana del último contacto")
    campaign: int = Field(..., ge=1, le=60, description="Contactos en esta campaña")
    pdays: int = Field(
        ...,
        ge=0,
        le=999,
        description="Días desde el contacto anterior; 999 = nunca contactado",
    )
    previous: int = Field(..., ge=0, le=50, description="Contactos en campañas anteriores")
    poutcome: PoutcomeType = Field(..., description="Resultado de la campaña anterior")

    # --- Contexto macroeconómico ---
    # Los alias con punto reproducen los nombres originales del dataset. Se
    # mantienen para que el payload de la API sea idéntico al CSV de origen y no
    # haya que traducir nombres entre capas.
    emp_var_rate: float = Field(
        ..., alias="emp.var.rate", ge=-5, le=5, description="Tasa de variación del empleo"
    )
    cons_price_idx: float = Field(
        ..., alias="cons.price.idx", ge=85, le=100, description="Índice de precios al consumidor"
    )
    cons_conf_idx: float = Field(
        ..., alias="cons.conf.idx", ge=-60, le=0, description="Índice de confianza del consumidor"
    )
    euribor3m: float = Field(..., ge=0, le=10, description="Euríbor a 3 meses")
    nr_employed: float = Field(
        ..., alias="nr.employed", ge=4000, le=6000, description="Número de empleados (miles)"
    )

    @field_validator("pdays")
    @classmethod
    def check_pdays_sentinel(cls, value: int) -> int:
        """
        `pdays` solo acepta días reales (0-40) o el centinela 999.

        Los valores intermedios no existen en el dataset y casi siempre son un
        error de codificación en el sistema de origen.
        """
        if value != 999 and value > 40:
            raise ValueError(
                "pdays debe estar entre 0 y 40 días, o ser exactamente 999 "
                "(nunca contactado antes)"
            )
        return value

    def to_model_input(self) -> dict[str, object]:
        """Convierte el modelo a las claves que espera el pipeline de sklearn."""
        return self.model_dump(by_alias=True)


class PredictionResponse(BaseModel):
    """Respuesta de una predicción individual."""

    probability: float = Field(..., description="Probabilidad de suscripción (0-1)")
    will_subscribe: bool = Field(..., description="Decisión al umbral vigente")
    propensity_band: str = Field(..., description="muy_baja | baja | media | alta")
    recommended_action: str = Field(..., description="llamar | no_llamar")
    threshold_used: float = Field(..., description="Umbral de decisión aplicado")
    model_name: str = Field(..., description="Nombre del modelo registrado")
    model_version: str = Field(..., description="Versión servida")
    predicted_at: str = Field(..., description="Timestamp UTC de la predicción")


class BatchPredictionRequest(BaseModel):
    """Petición de predicción para varios clientes."""

    model_config = ConfigDict(extra="forbid")

    clients: list[ClientFeatures] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Clientes a puntuar (máximo 1000 por petición)",
    )


class BatchPredictionItem(BaseModel):
    """Una fila del resultado batch."""

    index: int
    probability: float
    will_subscribe: bool
    propensity_band: str


class BatchPredictionResponse(BaseModel):
    """Respuesta de una predicción batch, con estadísticas agregadas."""

    predictions: list[BatchPredictionItem]
    n_clients: int
    n_to_call: int
    mean_probability: float
    threshold_used: float
    model_name: str
    model_version: str
    predicted_at: str


class HealthResponse(BaseModel):
    """Respuesta del health check."""

    status: Literal["healthy", "degraded"]
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    api_version: str
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    """Metadatos del modelo servido."""

    model_config = ConfigDict(extra="allow")

    name: str
    version: str
    alias: str
    run_id: str
    decision_threshold: float
    metrics: dict[str, float]
    loaded_at: str
    feature_columns: list[str]
