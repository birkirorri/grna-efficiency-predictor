from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    sequence: str = Field(
        ...,
        min_length=30,
        max_length=30,
        description="30-nucleotide sequence (4nt context + 20nt gRNA + 3nt PAM + 3nt context)"
    )


class PredictionResponse(BaseModel):
    sequence: str
    efficiency_score: float
    category: str
    model_version: str