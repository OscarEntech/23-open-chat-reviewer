from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import psycopg
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chatreview import __version__
from chatreview.config import Settings
from chatreview.db import Row, Session, database, migrate
from chatreview.embedding_models import (
    DEFAULT_EMBEDDING_PRESET,
    EmbeddingModelDependencyMissing,
    EmbeddingModelDownloadActive,
    EmbeddingModelManager,
    UnknownEmbeddingPreset,
)
from chatreview.episodes import episode_stats, get_episode, list_episodes
from chatreview.exporter import collect_evidence, render_evidence
from chatreview.registry import (
    delete_contributor_rule,
    list_activities,
    list_contributor_rules,
    list_contributors,
    list_project_aliases,
    list_projects,
    rebuild_registry,
    save_activity,
    save_contributor,
    save_contributor_rule,
    save_project_alias,
    set_occurrence_assignment,
    set_project_default_activity,
    unresolved_queues,
)
from chatreview.resume import list_resume_surfaces
from chatreview.search import (
    SearchFilters,
    corpus_stats,
    get_event,
    lexical_search,
    list_sessions,
    read_raw_event,
    reciprocal_rank_fusion,
    session_events,
)
from chatreview.semantic import SemanticSearchService, list_semantic_runs, map_points
from chatreview.setup_jobs import (
    BuildAlreadyRunning,
    InvalidBuildPlan,
    SetupBuildManager,
    SetupBuildPlan,
)
from chatreview.setup_planner import (
    HistoryScope as SetupHistoryScope,
)
from chatreview.setup_planner import (
    SemanticPolicy as SetupSemanticPolicy,
)
from chatreview.setup_planner import (
    SetupPlan,
    SetupPlanner,
)
from chatreview.summary_jobs import (
    SummaryAgentId,
    SummaryJobAlreadyRunning,
    SummaryRunManager,
    SummaryRunPlan,
    load_summary_agent,
    save_summary_agent,
    summary_agent_catalog,
)
from chatreview.timesheets import (
    TimesheetFilters,
    build_timesheet,
    compute_combined_timesheet,
    export_timesheet,
    latest_snapshot,
    list_timesheet_rows,
    timesheet_calendar,
    work_trail,
)
from chatreview.trace import build_session_trace


class AnnotationInput(BaseModel):
    target_type: Literal["session", "event", "window", "episode"]
    target_key: str = Field(min_length=1, max_length=256)
    label: str | None = Field(default=None, max_length=80)
    note: str | None = Field(default=None, max_length=100_000)
    review_state: Literal["unreviewed", "reviewing", "reviewed"] = "unreviewed"


class LabelInput(BaseModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[\w.-]+$")
    color: str = Field(default="#64748b", pattern=r"^#[0-9a-fA-F]{6}$")
    description: str | None = Field(default=None, max_length=500)


class ActivityInput(BaseModel):
    code: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")
    title: str = Field(min_length=1, max_length=300)
    classification: Literal["core", "supporting", "non-project", "unclassified"] = "unclassified"
    reporting_period_start: date
    reporting_period_end: date
    description: str | None = Field(default=None, max_length=20_000)
    uncertainty_or_hypothesis: str | None = Field(default=None, max_length=20_000)


class ProjectDefaultActivityInput(BaseModel):
    activity_id: int
    effective_from: datetime
    effective_to: datetime | None = None


class ProjectAliasInput(BaseModel):
    project_id: int
    path_prefix: str = Field(min_length=1, max_length=4096)
    machine_id: str | None = None
    provider: Literal["codex", "claude", "gemini", "git"] | None = None
    alias: str | None = Field(default=None, max_length=4096)
    effective_from: datetime | None = None
    effective_to: datetime | None = None


class OccurrenceAssignmentInput(BaseModel):
    activity_id: int
    project_id: int | None = None
    note: str | None = Field(default=None, max_length=20_000)


class ContributorInput(BaseModel):
    key: str = Field(min_length=1, max_length=200)
    display_name: str = Field(min_length=1, max_length=200)
    email: str | None = Field(default=None, max_length=320)


class ContributorRuleInput(BaseModel):
    machine_id: str
    contributor_id: int
    path_prefix: str | None = Field(default=None, max_length=4_096)
    provider: str | None = Field(default=None, max_length=80)
    effective_from: datetime | None = None
    effective_to: datetime | None = None


class CombinedTimesheetInput(BaseModel):
    financial_year: str = Field(pattern=r"^(?:FY)?\d{4}-(?:\d{2}|\d{4})$")
    project_keys: list[str] = Field(default_factory=list, max_length=500)


class SetupPreviewInput(BaseModel):
    history_start: date | None = None
    history_end: date | None = None
    providers: list[Literal["codex", "claude", "gemini"]] = Field(
        default_factory=lambda: ["codex", "claude", "gemini"]
    )
    include_git_metadata: bool = True
    preserve_encrypted_reasoning: bool = True
    include_readable_reasoning_in_search: bool = False
    include_reasoning_in_projection: bool = False
    embedding_preset: Literal["qwen3-embedding-0.6b"] = DEFAULT_EMBEDDING_PRESET


class SummaryAgentInput(BaseModel):
    provider: SummaryAgentId


class SummaryRunInput(SummaryAgentInput):
    days: int = Field(default=30, ge=1, le=365)
    limit: int = Field(default=40, ge=1, le=100)
    per_project_limit: int = Field(default=3, ge=1, le=10)


def create_app(settings: Settings) -> FastAPI:
    settings.ensure_output_dirs()
    migrate(settings.database_url)
    app = FastAPI(
        title="Open Chat Reviewer",
        version=__version__,
        description="Self-hosted PostgreSQL archive for Codex, Claude, Gemini, and Git evidence",
    )
    semantic = SemanticSearchService(settings)
    setup = SetupPlanner(settings)
    setup_builds = SetupBuildManager(settings)
    summary_runs = SummaryRunManager(settings)
    embedding_models = EmbeddingModelManager(settings.data_dir)

    def public_build_status() -> dict[str, Any]:
        status = setup_builds.status().to_dict()
        status["startedAt"] = status.pop("started_at")
        status["updatedAt"] = status.pop("updated_at")
        status["finishedAt"] = status.pop("finished_at")
        return status

    @contextmanager
    def db() -> Iterator[Session]:
        with database(settings.database_url) as connection:
            yield connection

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        with db() as connection:
            run = semantic.available_run(connection)
            return {
                "status": "ok",
                "version": __version__,
                "database": "PostgreSQL",
                "semantic_available": run is not None,
                "semantic_run": run["run_key"] if run else None,
                "semantic_freshness": run["freshness"] if run else None,
            }

    @app.get("/api/setup/status")
    def setup_status() -> dict[str, Any]:
        return setup.status().to_dict()

    @app.get("/api/setup/machines")
    def setup_machines() -> dict[str, Any]:
        return setup.machines().to_dict()

    @app.get("/api/setup/connection")
    def setup_connection() -> dict[str, Any]:
        return setup.connection().to_dict()

    @app.post("/api/setup/preview")
    def setup_preview(payload: SetupPreviewInput) -> dict[str, Any]:
        providers = [*payload.providers]
        if payload.include_git_metadata:
            providers.append("git")
        try:
            plan = SetupPlan(
                providers=tuple(providers),
                history=SetupHistoryScope(
                    start=payload.history_start,
                    end=payload.history_end,
                ),
                semantic=SetupSemanticPolicy(
                    include_reasoning=payload.include_reasoning_in_projection,
                    include_encoded_reasoning=False,
                    include_tool_calls=False,
                    include_context_summaries=False,
                ),
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        result = setup.preview(plan).to_dict()
        result["retention"] = {
            "preserve_encrypted_reasoning": payload.preserve_encrypted_reasoning,
            "include_readable_reasoning_in_search": (
                payload.include_readable_reasoning_in_search
            ),
            "include_reasoning_in_projection": payload.include_reasoning_in_projection,
        }
        return result

    @app.get("/api/setup/build")
    def setup_build_status() -> dict[str, Any]:
        return public_build_status()

    @app.post("/api/setup/build", status_code=202)
    def start_setup_build(payload: SetupPreviewInput) -> dict[str, Any]:
        try:
            setup_builds.start(
                SetupBuildPlan(
                    providers=tuple(payload.providers),
                    include_git=payload.include_git_metadata,
                    history_since=payload.history_start,
                    history_until=payload.history_end,
                    preserve_encrypted_reasoning=payload.preserve_encrypted_reasoning,
                    include_readable_reasoning_in_search=(
                        payload.include_readable_reasoning_in_search
                    ),
                    include_reasoning_in_projection=payload.include_reasoning_in_projection,
                    embedding_preset=payload.embedding_preset,
                    run_semantic_refresh=True,
                )
            )
        except BuildAlreadyRunning as exc:
            raise HTTPException(409, str(exc)) from exc
        except InvalidBuildPlan as exc:
            raise HTTPException(422, str(exc)) from exc
        return public_build_status()

    @app.delete("/api/setup/build")
    def cancel_setup_build() -> dict[str, Any]:
        setup_builds.cancel()
        return public_build_status()

    @app.get("/api/setup/embedding-models")
    def setup_embedding_models() -> list[dict[str, Any]]:
        return embedding_models.catalog()

    @app.post("/api/setup/embedding-models/{preset_id}/download", status_code=202)
    def download_setup_embedding_model(preset_id: str) -> dict[str, Any]:
        try:
            return embedding_models.start(preset_id)
        except UnknownEmbeddingPreset as exc:
            raise HTTPException(404, str(exc)) from exc
        except EmbeddingModelDownloadActive as exc:
            raise HTTPException(409, str(exc)) from exc
        except EmbeddingModelDependencyMissing as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/api/summary-agent")
    def summary_agent_status() -> dict[str, Any]:
        with database(settings.database_url, read_only=True) as connection:
            latest = list_resume_surfaces(connection, limit=1).get("latest_run")
        run = summary_runs.status()
        if not run["active"] and latest and latest.get("status") == "running":
            run = {
                "status": "running",
                "active": True,
                "provider": load_summary_agent(settings.data_dir),
                "message": (
                    f"Summarizing {int(latest.get('selected_count') or 0)} selected work threads"
                ),
                "started_at": latest.get("started_at"),
                "result": None,
            }
        return {
            "selected": load_summary_agent(settings.data_dir),
            "providers": summary_agent_catalog(),
            "run": run,
            "latest_run": latest,
        }

    @app.put("/api/summary-agent")
    def select_summary_agent(payload: SummaryAgentInput) -> dict[str, Any]:
        save_summary_agent(settings.data_dir, payload.provider)
        return {"selected": payload.provider}

    @app.post("/api/summary-agent/run", status_code=202)
    def start_summary_run(payload: SummaryRunInput) -> dict[str, Any]:
        with database(settings.database_url, read_only=True) as connection:
            latest = list_resume_surfaces(connection, limit=1).get("latest_run")
        if latest and latest.get("status") == "running":
            raise HTTPException(409, "a summary refresh is already running")
        try:
            return summary_runs.start(
                SummaryRunPlan(
                    provider=payload.provider,
                    days=payload.days,
                    limit=payload.limit,
                    per_project_limit=payload.per_project_limit,
                )
            )
        except SummaryJobAlreadyRunning as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        with db() as connection:
            return corpus_stats(connection)

    @app.get("/api/resume-surfaces")
    def resume_surfaces(
        limit: Annotated[int, Query(ge=1, le=500)] = 200,
    ) -> dict[str, Any]:
        with db() as connection:
            return list_resume_surfaces(connection, limit=limit)

    @app.get("/api/sources")
    def sources(
        provider: str | None = None,
        status: str | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        parameters: list[Any] = []
        if provider:
            clauses.append("source.provider=?")
            parameters.append(provider)
        if status:
            clauses.append("revision.status=?")
            parameters.append(status)
        parameters.extend([limit, offset])
        with db() as connection:
            rows = connection.execute(
                f"""
                SELECT source.id, source.provider, source.path, source.source_kind,
                       source.machine_id, machine.name AS machine_name,
                       revision.id AS revision_id, revision.revision_no,
                       revision.size_bytes, revision.ingested_offset,
                       revision.ingested_lines, revision.pending_length,
                       revision.status, revision.error_count, revision.last_error,
                       revision.aggregate_hash, revision.completed_at
                FROM sources source
                JOIN source_revisions revision ON revision.id=source.active_revision_id
                JOIN machines machine ON machine.id=source.machine_id
                WHERE {" AND ".join(clauses)}
                ORDER BY source.provider, source.path LIMIT ? OFFSET ?
                """,
                parameters,
            ).fetchall()
            return [_row(row) for row in rows]

    @app.get("/api/projects")
    def projects() -> list[dict[str, Any]]:
        with db() as connection:
            return list_projects(connection)

    @app.get("/api/project-aliases")
    def project_aliases() -> list[dict[str, Any]]:
        with db() as connection:
            return list_project_aliases(connection)

    @app.post("/api/project-aliases", status_code=201)
    def create_project_alias(payload: ProjectAliasInput) -> dict[str, Any]:
        with db() as connection:
            try:
                return save_project_alias(connection, **payload.model_dump())
            except psycopg.IntegrityError as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.post("/api/projects/rebuild", status_code=202)
    def rebuild_projects() -> dict[str, Any]:
        with db() as connection:
            summary = rebuild_registry(connection)
            return {
                "projects": summary.projects,
                "aliases": summary.aliases,
                "linked_sessions": summary.linked_sessions,
                "unresolved_projects": summary.unresolved_projects,
                "unresolved_activities": summary.unresolved_activities,
                "unresolved_contributors": summary.unresolved_contributors,
            }

    @app.get("/api/activities")
    def activities() -> list[dict[str, Any]]:
        with db() as connection:
            return list_activities(connection)

    @app.put("/api/activities/{code}")
    def upsert_activity(code: str, payload: ActivityInput) -> dict[str, Any]:
        if code != payload.code:
            raise HTTPException(422, "path and payload activity codes differ")
        with db() as connection:
            try:
                return save_activity(connection, **payload.model_dump())
            except psycopg.IntegrityError as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.put("/api/projects/{project_id}/default-activity")
    def project_default_activity(
        project_id: int, payload: ProjectDefaultActivityInput
    ) -> dict[str, Any]:
        with db() as connection:
            try:
                return set_project_default_activity(
                    connection,
                    project_id=project_id,
                    activity_id=payload.activity_id,
                    effective_from=payload.effective_from.isoformat(),
                    effective_to=(
                        payload.effective_to.isoformat() if payload.effective_to else "infinity"
                    ),
                )
            except psycopg.IntegrityError as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.put("/api/occurrences/{episode_key}/activity")
    def occurrence_activity(
        episode_key: str, payload: OccurrenceAssignmentInput
    ) -> dict[str, Any]:
        with db() as connection:
            try:
                return set_occurrence_assignment(
                    connection,
                    episode_key=episode_key,
                    **payload.model_dump(),
                )
            except KeyError as exc:
                raise HTTPException(404, "occurrence not found") from exc
            except psycopg.IntegrityError as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.get("/api/registry/unresolved")
    def registry_unresolved() -> dict[str, list[dict[str, Any]]]:
        with db() as connection:
            return unresolved_queues(connection)

    @app.get("/api/contributors")
    def contributors() -> list[dict[str, Any]]:
        with db() as connection:
            return list_contributors(connection)

    @app.put("/api/contributors/{key}")
    def upsert_contributor(key: str, payload: ContributorInput) -> dict[str, Any]:
        if key != payload.key:
            raise HTTPException(422, "path and payload contributor keys differ")
        with db() as connection:
            return save_contributor(connection, **payload.model_dump())

    @app.get("/api/contributor-rules")
    def contributor_rules() -> list[dict[str, Any]]:
        with db() as connection:
            return list_contributor_rules(connection)

    @app.post("/api/contributor-rules", status_code=201)
    def create_contributor_rule(payload: ContributorRuleInput) -> dict[str, Any]:
        with db() as connection:
            try:
                return save_contributor_rule(connection, **payload.model_dump())
            except psycopg.IntegrityError as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.delete("/api/contributor-rules/{rule_id}", status_code=204)
    def remove_contributor_rule(rule_id: int) -> Response:
        with db() as connection:
            if not delete_contributor_rule(connection, rule_id):
                raise HTTPException(404, "contributor rule not found")
        return Response(status_code=204)

    @app.get("/api/timeline")
    def timeline(
        provider: str | None = None,
        project: str | None = None,
        bucket: Literal["day", "week", "month"] = "day",
    ) -> list[dict[str, Any]]:
        format_by_bucket = {
            "day": "YYYY-MM-DD",
            "week": 'IYYY-"W"IW',
            "month": "YYYY-MM",
        }
        clauses = ["e.timestamp IS NOT NULL"]
        parameters: list[Any] = []
        if provider:
            clauses.append("s.provider=?")
            parameters.append(provider)
        if project:
            clauses.append("s.project=?")
            parameters.append(project)
        with db() as connection:
            rows = connection.execute(
                f"""
                SELECT to_char(date_trunc('{bucket}', e.timestamp), '{format_by_bucket[bucket]}') AS bucket,
                       s.provider, COUNT(*) AS event_count,
                       COUNT(DISTINCT e.session_id) AS session_count
                FROM events e JOIN sessions s ON s.id=e.session_id
                WHERE {" AND ".join(clauses)}
                GROUP BY bucket, s.provider ORDER BY bucket
                """,
                parameters,
            ).fetchall()
            return [_row(row) for row in rows]

    @app.get("/api/sessions")
    def sessions(
        provider: str | None = None,
        project: str | None = None,
        q: str | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> list[dict[str, Any]]:
        with db() as connection:
            return list_sessions(
                connection,
                provider=provider,
                project=project,
                query=q,
                limit=limit,
                offset=offset,
            )

    @app.get("/api/sessions/{session_id}")
    def session_detail(session_id: int) -> dict[str, Any]:
        with db() as connection:
            row = connection.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                raise HTTPException(404, "session not found")
            result = _row(row)
            result["metadata"] = _json(result.pop("metadata_json", "{}"))
            result["annotations"] = _annotations(connection, "session", result["session_key"])
            return result

    @app.get("/api/sessions/{session_id}/events")
    def events(
        session_id: int,
        limit: Annotated[int, Query(ge=1, le=5000)] = 500,
        offset: Annotated[int, Query(ge=0)] = 0,
        include_empty: bool = False,
    ) -> list[dict[str, Any]]:
        with db() as connection:
            return session_events(
                connection, session_id, limit=limit, offset=offset, include_empty=include_empty
            )

    @app.get("/api/sessions/{session_id}/trace")
    def session_trace(
        session_id: int,
        occurrence_limit: Annotated[int, Query(ge=1, le=500)] = 500,
        run_limit: Annotated[int, Query(ge=10, le=500)] = 120,
    ) -> dict[str, Any]:
        with db() as connection:
            trace = build_session_trace(
                connection,
                session_id,
                occurrence_limit=occurrence_limit,
                run_limit=run_limit,
            )
            if trace is None:
                raise HTTPException(404, "session not found")
            return trace

    @app.get("/api/episodes/stats")
    def episodes_stats() -> dict[str, Any]:
        with db() as connection:
            return episode_stats(connection)

    @app.get("/api/episodes")
    def episodes(
        q: str | None = None,
        provider: str | None = None,
        project: str | None = None,
        evidence_state: str | None = None,
        errors_only: bool = False,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> list[dict[str, Any]]:
        with db() as connection:
            return list_episodes(
                connection,
                query=q,
                provider=provider,
                project=project,
                evidence_state=evidence_state,
                errors_only=errors_only,
                limit=limit,
                offset=offset,
            )

    @app.get("/api/episodes/{episode_id}")
    def episode_detail(episode_id: int) -> dict[str, Any]:
        with db() as connection:
            episode = get_episode(connection, episode_id)
            if episode is None:
                raise HTTPException(404, "episode not found")
            episode["annotations"] = _annotations(connection, "episode", episode["episode_key"])
            return episode

    @app.get("/api/events/{event_id}")
    def event_detail(event_id: int) -> dict[str, Any]:
        with db() as connection:
            event = get_event(connection, event_id)
            if event is None:
                raise HTTPException(404, "event not found")
            event["annotations"] = _annotations(connection, "event", event["event_key"])
            return event

    @app.get("/api/events/{event_id}/raw")
    def raw_event(
        event_id: int,
        full: bool = False,
        as_text: bool = False,
    ) -> Any:
        with db() as connection:
            raw = read_raw_event(connection, event_id, max_bytes=None if full else 2_000_000)
            if raw is None:
                raise HTTPException(404, "event not found")
            if as_text and raw.get("raw") is not None:
                media_type = (
                    "application/json"
                    if raw.get("provider") == "gemini"
                    else "application/x-ndjson"
                )
                return Response(raw["raw"], media_type=media_type)
            return raw

    @app.get("/api/search")
    def search(
        q: Annotated[str, Query(min_length=1, max_length=10_000)],
        mode: Literal["lexical", "semantic", "hybrid"] = "hybrid",
        provider: str | None = None,
        project: str | None = None,
        contributor: str | None = None,
        activity: str | None = None,
        activity_classification: Literal["core", "supporting", "non-project", "unclassified"] | None = None,
        role: str | None = None,
        kind: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        errors_only: bool = False,
        include_reasoning: bool | None = None,
        profile: Literal["conversation", "episodes"] | None = None,
        run_key: str | None = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        if date_from and date_to and date_from > date_to:
            raise HTTPException(422, "date_from must be on or before date_to")
        if include_reasoning is None:
            saved_plan = setup_builds.status().plan
            include_reasoning = bool(
                saved_plan.get("include_readable_reasoning_in_search", False)
            )
        filters = SearchFilters(
            provider=provider,
            project=project,
            contributor=contributor,
            activity=activity,
            activity_classification=activity_classification,
            role=role,
            kind=kind,
            date_from=date_from.isoformat() if date_from else None,
            date_to=date_to.isoformat() if date_to else None,
            errors_only=errors_only,
            include_reasoning=include_reasoning,
        )
        with db() as connection:
            candidate_limit = 100 if mode == "hybrid" else limit
            lexical = (
                lexical_search(
                    connection,
                    q,
                    filters=filters,
                    limit=candidate_limit,
                    offset=offset if mode == "lexical" else 0,
                )
                if mode in {"lexical", "hybrid"}
                else []
            )
            semantic_results: list[dict[str, Any]] = []
            semantic_error = None
            if mode in {"semantic", "hybrid"}:
                try:
                    semantic_results = semantic.search(
                        connection,
                        q,
                        filters=filters,
                        limit=candidate_limit,
                        profile=profile,
                        run_key=run_key,
                    )
                except RuntimeError as exc:
                    semantic_error = str(exc)
            hits = (
                reciprocal_rank_fusion(lexical, semantic_results, limit=limit + offset)[offset:]
                if mode == "hybrid"
                else lexical or semantic_results
            )
            return {
                "query": q,
                "mode": mode,
                "hits": _public_search_hits(hits),
                "lexical": _public_search_hits(lexical[:limit]),
                "semantic": _public_search_hits(semantic_results[:limit]),
                "semantic_error": semantic_error,
                "semantic_profile": profile,
                "semantic_run_key": run_key,
            }

    @app.get("/api/archive-status")
    def archive_status() -> dict[str, Any]:
        with db() as connection:
            row = connection.execute(
                """
                SELECT (SELECT COUNT(*) FROM sources) AS sources,
                       (SELECT COUNT(*) FROM source_revisions) AS revisions,
                       (SELECT COUNT(*) FROM raw_records) AS raw_records,
                       (SELECT COUNT(*) FROM raw_payloads) AS raw_payloads,
                       (
                           SELECT COUNT(*) FROM raw_payloads
                           WHERE byte_length<>octet_length(payload)
                       ) AS length_mismatches,
                       (
                           SELECT COUNT(*) FROM sources source
                           JOIN source_revisions revision ON revision.id=source.active_revision_id
                           WHERE revision.status IN ('failed', 'partial')
                              OR revision.pending_length>0
                       ) AS attention_revisions,
                       (
                           SELECT COALESCE(SUM(revision.pending_length), 0)::bigint
                           FROM sources source
                           JOIN source_revisions revision ON revision.id=source.active_revision_id
                       ) AS pending_bytes
                """
            ).fetchone()
            statuses = {
                item["status"]: int(item["count"])
                for item in connection.execute(
                    """
                    SELECT revision.status, COUNT(*) AS count
                    FROM sources source
                    JOIN source_revisions revision ON revision.id=source.active_revision_id
                    GROUP BY revision.status
                    """
                )
            }
            result = _row(row)
            result["statuses"] = statuses
            result["privacy"] = "Raw chats remain in the self-hosted PostgreSQL archive."
            return result

    @app.get("/api/work-trail")
    def categorized_work_trail(
        project: str | None = None,
        activity: str | None = None,
        classification: Literal["core", "supporting", "non-project", "unclassified"] | None = None,
        limit: Annotated[int, Query(ge=1, le=2000)] = 200,
    ) -> list[dict[str, Any]]:
        with db() as connection:
            return work_trail(
                connection,
                project=project,
                activity=activity,
                classification=classification,
                limit=limit,
            )

    @app.post("/api/timesheets/build", status_code=202)
    def build_timesheet_snapshot(cutoff: datetime | None = None, force: bool = False) -> dict[str, Any]:
        with db() as connection:
            summary = build_timesheet(connection, cutoff=cutoff, force=force)
            return {
                "snapshot_id": summary.snapshot_id,
                "snapshot_key": summary.snapshot_key,
                "intervals": summary.intervals,
                "total_seconds": summary.total_seconds,
                "ambiguity_count": summary.ambiguity_count,
                "corpus_fingerprint": summary.corpus_fingerprint,
                "cutoff": summary.cutoff,
                "reused": summary.reused,
            }

    @app.get("/api/timesheets")
    def timesheet_rows(
        date_from: date | None = None,
        date_to: date | None = None,
        contributor: str | None = None,
        project: str | None = None,
        projects: Annotated[list[str] | None, Query()] = None,
        activity: str | None = None,
        classification: Literal["core", "supporting", "non-project", "unclassified"] | None = None,
        limit: Annotated[int, Query(ge=1, le=2000)] = 500,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        filters = TimesheetFilters(
            date_from=date_from,
            date_to=date_to,
            contributor=contributor,
            project=project,
            projects=tuple(projects or ()),
            activity=activity,
            classification=classification,
        )
        with db() as connection:
            return {
                "snapshot": latest_snapshot(connection),
                "rows": list_timesheet_rows(
                    connection, filters=filters, limit=limit, offset=offset
                ),
            }

    @app.get("/api/timesheets/calendar")
    def timesheet_calendar_view(
        financial_year: Annotated[
            str | None,
            Query(pattern=r"^(?:FY)?\d{4}-(?:\d{2}|\d{4})$"),
        ] = None,
        year: Annotated[int | None, Query(ge=2000, le=2100)] = None,
    ) -> dict[str, Any]:
        if financial_year is not None and year is not None:
            raise HTTPException(422, "choose financial_year or year, not both")
        with db() as connection:
            try:
                return timesheet_calendar(
                    connection,
                    financial_year=financial_year,
                    year=year,
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.post("/api/timesheets/compute")
    def compute_timesheet(payload: CombinedTimesheetInput) -> dict[str, Any]:
        with db() as connection:
            try:
                return compute_combined_timesheet(
                    connection,
                    financial_year=payload.financial_year,
                    project_keys=tuple(payload.project_keys),
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.get("/api/timesheets/export")
    def timesheet_export(
        format: Literal["csv", "markdown", "json"] = "csv",
        date_from: date | None = None,
        date_to: date | None = None,
        contributor: str | None = None,
        project: str | None = None,
        activity: str | None = None,
        classification: Literal["core", "supporting", "non-project", "unclassified"] | None = None,
    ) -> Response:
        filters = TimesheetFilters(
            date_from=date_from,
            date_to=date_to,
            contributor=contributor,
            project=project,
            activity=activity,
            classification=classification,
        )
        with db() as connection:
            try:
                result = export_timesheet(connection, format=format, filters=filters)
            except ValueError as exc:
                raise HTTPException(404, str(exc)) from exc
        media_type = {
            "csv": "text/csv; charset=utf-8",
            "markdown": "text/markdown; charset=utf-8",
            "json": "application/json",
        }[format]
        return Response(
            content=result.content,
            media_type=media_type,
            headers={"X-ChatReview-Manifest-SHA256": result.manifest["manifest_sha256"]},
        )

    @app.get("/api/semantic-runs")
    def semantic_runs() -> list[dict[str, Any]]:
        with db() as connection:
            return list_semantic_runs(connection)

    @app.get("/api/map")
    def semantic_map(
        run_id: int | None = None,
        profile: Literal["conversation", "episodes"] | None = None,
        provider: str | None = None,
        project: str | None = None,
        cluster_id: int | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        recent_days: Annotated[int | None, Query(ge=1, le=3650)] = 30,
        limit: Annotated[int, Query(ge=100, le=100_000)] = 20_000,
    ) -> dict[str, Any]:
        if date_from and date_to and date_from > date_to:
            raise HTTPException(422, "date_from must be on or before date_to")
        with db() as connection:
            return map_points(
                connection,
                run_id=run_id,
                profile=profile,
                provider=provider,
                project=project,
                cluster_id=cluster_id,
                date_from=date_from,
                date_to=date_to,
                recent_days=recent_days,
                limit=limit,
            )

    @app.get("/api/windows/{window_id}")
    def window_detail(window_id: int) -> dict[str, Any]:
        with db() as connection:
            row = connection.execute(
                """
                SELECT w.*, c.text, s.session_key, s.external_id, s.provider, s.project
                FROM semantic_windows w JOIN contents c ON c.id=w.content_id
                JOIN sessions s ON s.id=w.session_id WHERE w.id=?
                """,
                (window_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(404, "window not found")
            result = _row(row)
            result["annotations"] = _annotations(connection, "window", result["window_key"])
            return result

    @app.get("/api/artifacts")
    def artifacts(
        kind: str | None = None,
        q: str | None = None,
        session_id: int | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        parameters: list[Any] = []
        if kind:
            clauses.append("a.kind=?")
            parameters.append(kind)
            if kind == "error-signature":
                clauses.append("is_actionable_error_signature(a.value)=1")
        if q:
            clauses.append("a.value ILIKE ?")
            parameters.append(f"%{q}%")
        if session_id is not None:
            clauses.append("e.session_id=?")
            parameters.append(session_id)
        parameters.extend([limit, offset])
        with db() as connection:
            rows = connection.execute(
                f"""
                SELECT a.id, a.kind, a.label, a.value, a.value_hash, e.id AS event_id,
                       e.event_key, e.timestamp, s.id AS session_id, s.provider, s.project
                FROM artifacts a JOIN events e ON e.id=a.event_id
                LEFT JOIN sessions s ON s.id=e.session_id
                WHERE {" AND ".join(clauses)} ORDER BY e.timestamp DESC
                LIMIT ? OFFSET ?
                """,
                parameters,
            ).fetchall()
            return [_row(row) for row in rows]

    @app.get("/api/labels")
    def labels() -> list[dict[str, Any]]:
        with db() as connection:
            rows = connection.execute(
                """
                SELECT l.*, COUNT(a.id) AS annotation_count FROM labels l
                LEFT JOIN annotations a ON a.label_id=l.id GROUP BY l.id ORDER BY l.name
                """
            ).fetchall()
            return [_row(row) for row in rows]

    @app.post("/api/labels", status_code=201)
    def create_label(payload: LabelInput) -> dict[str, Any]:
        with db() as connection:
            try:
                row = connection.execute(
                    """
                    INSERT INTO labels(name, color, description) VALUES (?, ?, ?)
                    RETURNING *
                    """,
                    (payload.name, payload.color, payload.description),
                ).fetchone()
                connection.commit()
            except psycopg.IntegrityError as exc:
                raise HTTPException(409, "label already exists") from exc
            assert row is not None
            return _row(row)

    @app.get("/api/annotations")
    def annotations(
        target_type: str | None = None,
        target_key: str | None = None,
        label: str | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 500,
    ) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        parameters: list[Any] = []
        if target_type:
            clauses.append("a.target_type=?")
            parameters.append(target_type)
        if target_key:
            clauses.append("a.target_key=?")
            parameters.append(target_key)
        if label:
            clauses.append("l.name=?")
            parameters.append(label)
        parameters.append(limit)
        with db() as connection:
            rows = connection.execute(
                f"""
                SELECT a.*, l.name AS label, l.color FROM annotations a
                LEFT JOIN labels l ON l.id=a.label_id WHERE {" AND ".join(clauses)}
                ORDER BY a.updated_at DESC LIMIT ?
                """,
                parameters,
            ).fetchall()
            return [_row(row) for row in rows]

    @app.post("/api/annotations", status_code=201)
    def save_annotation(payload: AnnotationInput) -> dict[str, Any]:
        with db() as connection:
            label_id = None
            if payload.label:
                label_row = connection.execute(
                    "SELECT id FROM labels WHERE name=?", (payload.label,)
                ).fetchone()
                if label_row is None:
                    raise HTTPException(400, f"unknown label: {payload.label}")
                label_id = int(label_row["id"])
            connection.execute(
                """
                INSERT INTO annotations(target_type, target_key, label_id, note, review_state)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(target_type, target_key, label_id) DO UPDATE SET
                    note=excluded.note, review_state=excluded.review_state,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    payload.target_type,
                    payload.target_key,
                    label_id,
                    payload.note,
                    payload.review_state,
                ),
            )
            connection.commit()
            rows = _annotations(connection, payload.target_type, payload.target_key)
            return rows[-1] if rows else {}

    @app.delete("/api/annotations/{annotation_id}", status_code=204)
    def delete_annotation(annotation_id: int) -> Response:
        with db() as connection:
            cursor = connection.execute("DELETE FROM annotations WHERE id=?", (annotation_id,))
            connection.commit()
            if cursor.rowcount == 0:
                raise HTTPException(404, "annotation not found")
            return Response(status_code=204)

    @app.get("/api/export")
    def export(
        format: Literal["markdown", "jsonl", "csv"] = "markdown",
        q: str | None = None,
        session_id: int | None = None,
        label: str | None = None,
        limit: Annotated[int, Query(ge=1, le=5000)] = 1000,
    ) -> Response:
        with db() as connection:
            try:
                records = collect_evidence(
                    connection, query=q, session_id=session_id, label=label, limit=limit
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            output = render_evidence(records, format)
            media_types = {
                "markdown": "text/markdown",
                "jsonl": "application/x-ndjson",
                "csv": "text/csv",
            }
            return Response(output, media_type=media_types[format])

    @app.get("/token-report", include_in_schema=False)
    def token_report_panel() -> Response:
        """Render the token cost report as standalone HTML."""
        from chatreview.token_panel import render_html

        return Response(render_html(settings.database_url), media_type="text/html; charset=utf-8")

    web_dist = Path(__file__).resolve().parents[2] / "web" / "dist"
    if web_dist.is_dir():
        assets = web_dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            requested = web_dist / path
            if path and requested.is_file() and web_dist in requested.resolve().parents:
                return FileResponse(requested)
            return FileResponse(web_dist / "index.html")

    return app


def _public_search_hits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return bounded list-view fields; full evidence has dedicated detail routes."""

    return [{key: value for key, value in row.items() if key != "text"} for row in rows]


def _annotations(connection: Session, target_type: str, target_key: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT a.*, l.name AS label, l.color FROM annotations a
        LEFT JOIN labels l ON l.id=a.label_id
        WHERE a.target_type=? AND a.target_key=? ORDER BY a.created_at
        """,
        (target_type, target_key),
    ).fetchall()
    return [_row(row) for row in rows]


def _row(row: Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _json(value: Any) -> Any:
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
