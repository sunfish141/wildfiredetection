"""Local FastAPI map application for bounded, stateless fire rollouts."""

from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .incident_transition import EvidenceCell
from .live_firms import DEFAULT_BOUNDS, fetch_current_firms, LiveFirmsError
from .recursive_transition import ActiveFireCell, RecursiveFireState
from .scheduled_sampling import load_pass_model
from .terrain_features import TerrainFeatureSampler
from .training_grid import cell_from_id, cell_from_wgs84


ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).with_name("web")
DEFAULT_RUN = ROOT / "artifacts/incident-two-pass-v1-201db0d293c56f51/run_manifest.json"


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CellInput(Input):
    cell_id: str = Field(max_length=80)
    intensity: float = Field(ge=0, le=1)
    remaining_active_steps: int = Field(ge=1, le=2, strict=True)
    observation_age_hours: float = Field(ge=0, le=120)
    detection_count: int | None = Field(default=None, ge=1, le=100000)
    bright_ti4_max: float | None = None
    bright_ti4_mean: float | None = None
    platform_count: int | None = Field(default=None, ge=1, le=3)

    @field_validator("cell_id")
    @classmethod
    def valid_cell(cls, value):
        cell_from_id(value)
        return value

    def to_cell(self):
        fields = self.model_dump(exclude_none=True)
        evidence = [self.detection_count, self.bright_ti4_max, self.bright_ti4_mean, self.platform_count]
        if any(v is not None for v in evidence):
            if any(v is None for v in evidence):
                raise ValueError("FIRMS cells require all observation aggregates")
            return EvidenceCell(**fields)
        return ActiveFireCell(**fields)


class StateInput(Input):
    step_index: int = Field(ge=0, strict=True)
    active_cells: list[CellInput]
    burned_cell_ids: list[Annotated[str, Field(max_length=80)]] = Field(default_factory=list)

    def to_state(self):
        for cell_id in self.burned_cell_ids:
            CellInput.valid_cell(cell_id)
        return RecursiveFireState(self.step_index, tuple(c.to_cell() for c in self.active_cells), tuple(self.burned_cell_ids))


class StepInput(Input):
    state: StateInput
    origin_at: datetime

    @field_validator("origin_at")
    @classmethod
    def aware(cls, value):
        if value.tzinfo is None:
            raise ValueError("origin_at must include a timezone")
        return value.astimezone(timezone.utc)


class IgnitionInput(Input):
    latitude: float = Field(ge=24, le=84)
    longitude: float = Field(ge=-179, le=-50)
    intensity: float = Field(ge=0, le=1)


class SeedInput(Input):
    ignitions: list[IgnitionInput] = Field(min_length=1, max_length=500)


class BoundsInput(Input):
    west: float = Field(ge=-179, le=-50)
    south: float = Field(ge=24, le=84)
    east: float = Field(ge=-179, le=-50)
    north: float = Field(ge=24, le=84)

    @model_validator(mode="after")
    def bounded(self):
        if self.east <= self.west or self.north <= self.south:
            raise ValueError("Bounds must have east > west and north > south")
        return self


def state_response(state, *, origin_at, predictions=(), metadata=None, terrain_missing=0):
    active = {c.cell_id: c for c in state.active_cells}
    burned = set(state.burned_cell_ids)
    points = []
    scores = {p.cell_id: p for p in predictions}
    for cell_id in sorted(set(active) | burned | set(scores)):
        lat, lon = cell_from_id(cell_id).center_wgs84
        cell, score = active.get(cell_id), scores.get(cell_id)
        status = "active" if cell else "burned" if cell_id in burned else "candidate"
        points.append({"cell_id": cell_id, "latitude": lat, "longitude": lon, "status": status,
            "intensity": cell.intensity if cell else None,
            "ignition_probability": score.ignition_probability if score else None,
            "new_ignition": bool(score and score.will_ignite),
            "source": "FIRMS observation" if isinstance(cell, EvidenceCell) else "Placed ignition" if cell and state.step_index == 0 else "Simulation",
            "observation_age_hours": cell.observation_age_hours if cell else None,
            "detection_count": cell.detection_count if isinstance(cell, EvidenceCell) else None,
            "bright_ti4_max": cell.bright_ti4_max if isinstance(cell, EvidenceCell) else None,
            "remaining_active_steps": cell.remaining_active_steps if cell else None})
    return {"state": asdict(state), "origin_at": origin_at.isoformat(),
        "valid_at": (origin_at + timedelta(hours=12 * state.step_index)).isoformat(),
        "elapsed_hours": 12 * state.step_index, "points": points,
        "active_count": len(active), "burned_count": len(state.burned_cell_ids),
        "new_ignition_count": sum(p.will_ignite for p in predictions),
        "finished": False, "extinct": not active,
        "terrain_missing_count": terrain_missing, "metadata": metadata}


def create_app(*, model=None, terrain_provider=None, firms_loader=None):
    # Injection keeps behavioral tests independent of private local artifacts.
    load_dotenv(ROOT / "config/.env")
    api_key = os.getenv("NASA_FIRMS_API_KEY") or os.getenv("MAP_KEY") or ""
    pass_name = os.getenv("WILDFIRE_MODEL_PASS", "pass_2")
    inference_lock = Lock()
    firms_lock = Lock()
    firms_cache = {}

    @asynccontextmanager
    async def lifespan(app):
        app.state.model = model
        app.state.terrain = terrain_provider
        app.state.model_error = None
        try:
            if app.state.model is None:
                app.state.model = load_pass_model(Path(os.getenv("WILDFIRE_RUN_MANIFEST", str(DEFAULT_RUN))), pass_name)
            if app.state.terrain is None:
                sampler = TerrainFeatureSampler(Path(os.getenv("WILDFIRE_DATA_ROOT", str(ROOT / "data"))), max_cached_blocks=4)
                app.state.terrain = lru_cache(maxsize=8192)(sampler.sample_cell)
        except Exception:
            logging.getLogger(__name__).exception("Could not initialize the local wildfire model")
            app.state.model_error = "Model unavailable. Check WILDFIRE_RUN_MANIFEST and WILDFIRE_DATA_ROOT on the server."
        yield

    app = FastAPI(title="Wildfire Atlas", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def ready_model():
        if app.state.model_error:
            raise HTTPException(503, app.state.model_error)
        return app.state.model

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    def config():
        return {"model_ready": app.state.model_error is None, "model_error": app.state.model_error,
            "model_name": f"Incident model · {pass_name.replace('_', ' ')}", "research_preview": True,
            "firms_configured": bool(api_key or firms_loader), "step_hours": 12,
            "max_steps": None, "default_speed_seconds": 3, "max_seed_cells": 500,
            "firms_bounds": list(DEFAULT_BOUNDS)}

    @app.post("/api/seed")
    def seed(body: SeedInput):
        current = ready_model()
        ignitions = {}
        for point in body.ignitions:
            cell_id = cell_from_wgs84(latitude=point.latitude, longitude=point.longitude).cell_id
            ignitions[cell_id] = max(ignitions.get(cell_id, 0), point.intensity)
        state = current.initial_state(ignitions)
        return state_response(state, origin_at=datetime.now(timezone.utc))

    @app.post("/api/step")
    def step(body: StepInput):
        current = ready_model()
        try:
            state = body.state.to_state()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        missing = set()
        def terrain(cell_id):
            values = app.state.terrain(cell_id)
            if values.get("terrain_coverage_status") != "sampled":
                missing.add(cell_id)
            return values
        with inference_lock:
            result = current.step(state, terrain_provider=terrain)
        return state_response(result.state, origin_at=body.origin_at,
            predictions=result.predictions, terrain_missing=len(missing))

    @app.post("/api/firms")
    def firms(body: BoundsInput | None = None):
        ready_model()
        bounds = (body.west, body.south, body.east, body.north) if body else DEFAULT_BOUNDS
        # Repeated clicks in the same view reuse a preview for five minutes.
        with firms_lock:
            now = datetime.now(timezone.utc)
            cached = firms_cache.get(bounds)
            if cached and (now - cached[0]).total_seconds() < 300:
                return cached[1]
            try:
                state, metadata = (firms_loader or fetch_current_firms)(api_key, bounds, now=now)
            except LiveFirmsError as exc:
                raise HTTPException(502 if api_key or firms_loader else 503, str(exc)) from None
            response = state_response(state, origin_at=now, metadata=metadata)
            if len(firms_cache) >= 16:
                firms_cache.pop(next(iter(firms_cache)))
            firms_cache[bounds] = (now, response)
            return response

    return app


app = create_app()
