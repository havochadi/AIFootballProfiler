"""All-position interval review API; legacy whole-dataset reviews stay separate."""
from typing import Annotated
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictInt
from . import cases as C, learning as L, taxonomy as T

router = APIRouter(prefix='/api')
Percentage = Annotated[float, Field(ge=0, le=100)]


@router.get('/taxonomy')
def taxonomy():
    return T.CATALOGUE


@router.get('/cases/summary')
def summary():
    return {**C.summary(), 'agreement': L.agreement(T.VERSION)}


@router.get('/cases/export')
def export():
    return JSONResponse(C.export(), headers={'Content-Disposition': 'attachment; filename=interval_reviews_v2.json'})


@router.get('/cases')
def listing(dataset_id: str | None = None, player_id: str | None = None):
    return C.listing(dataset_id, player_id)


class Case(BaseModel):
    dataset_id: str
    player_id: str
    period: StrictInt = 1
    start_s: float = Field(ge=0, allow_inf_nan=False)
    end_s: float = Field(gt=0, allow_inf_nan=False)
    position_group: str
    context: str = Field(default='', max_length=3000)


@router.post('/cases')
def create(payload: Case):
    return C.create(**payload.model_dump())


@router.get('/cases/{case_id}')
def case(case_id: str):
    return C.get(case_id)


@router.get('/cases/{case_id}/assessment')
def assessment(case_id: str):
    case = C.get(case_id)
    prediction = ({'status': 'source_changed', 'message': 'Source data changed; create and review a new case.'}
                  if case['stale'] else L.predict(case['profile']))
    return {'consensus': C.consensus(case_id), 'prediction': prediction}


@router.get('/cases/{case_id}/reviews')
def reviews(case_id: str, reviewer: str = ''):
    C.get(case_id)
    rows = C.reviews(case_id)
    own = next((r for r in rows if r['reviewer'] == reviewer.strip().casefold()), None)
    return {'own_review': own, 'reviewer_count': len(rows)}


class Sequence(BaseModel):
    start_s: float = Field(ge=0, allow_inf_nan=False)
    end_s: float = Field(gt=0, allow_inf_nan=False)
    note: str = Field(min_length=1, max_length=2000)


class Review(BaseModel):
    reviewer: str = Field(min_length=1, max_length=80)
    labels: dict[str, Percentage | None]
    evidence: str = Field(min_length=1, max_length=3000)
    sequences: list[Sequence] = Field(default_factory=list, max_length=100)
    confidence: str = 'uncertain'
    abstain_reason: str = Field(default='', max_length=3000)
    notes: str = Field(default='', max_length=3000)


@router.post('/cases/{case_id}/review')
def review(case_id: str, payload: Review):
    return C.save_review(case_id, **payload.model_dump())


class Adjudication(BaseModel):
    label: str
    value: Percentage
    reviewer: str = Field(min_length=1, max_length=80)
    reason: str = Field(min_length=1, max_length=3000)


@router.post('/cases/{case_id}/adjudicate')
def adjudicate(case_id: str, payload: Adjudication):
    return C.adjudicate(case_id, **payload.model_dump())
