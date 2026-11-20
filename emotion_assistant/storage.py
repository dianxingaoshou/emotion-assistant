"""Thread-safe local SQLite records with consent enforcement and JSON export."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
import json
from pathlib import Path
import sqlite3
import secrets
import threading
import uuid

from .core import interval_duration, interval_union, utc_now
from .personalization import FeedbackRecord, build_reference, comparison_key, matches_final_revision
from .crossover import arm_for


class Store:
    def __init__(self, path):
        self.path = Path(path) if str(path) != ":memory:" else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                started_at TEXT NOT NULL, ended_at TEXT, consent_json TEXT NOT NULL,
                mode TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS observations (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                event_id TEXT NOT NULL, modality TEXT NOT NULL, t_start REAL NOT NULL,
                t_end REAL NOT NULL, valid INTEGER NOT NULL, payload_json TEXT NOT NULL,
                PRIMARY KEY(session_id, event_id)
            );
            CREATE TABLE IF NOT EXISTS utterances (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                id TEXT NOT NULL, payload_json TEXT NOT NULL,
                PRIMARY KEY(session_id, id)
            );
            CREATE TABLE IF NOT EXISTS pending_transcript_cleanup (
                session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE
            );
            CREATE TRIGGER IF NOT EXISTS no_utterance_after_revoke_request
                BEFORE INSERT ON utterances
                WHEN EXISTS (SELECT 1 FROM pending_transcript_cleanup WHERE session_id=NEW.session_id)
                BEGIN SELECT RAISE(ABORT,'transcript cleanup pending'); END;
            CREATE TABLE IF NOT EXISTS self_reports (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                phase TEXT NOT NULL, distress INTEGER NOT NULL CHECK(distress BETWEEN 0 AND 10),
                intervention_id TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS interventions (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                id TEXT NOT NULL, payload_json TEXT NOT NULL,
                PRIMARY KEY(session_id, id)
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                kind TEXT NOT NULL, created_at TEXT NOT NULL, payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS activity_preferences (
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                activity TEXT NOT NULL,
                preference TEXT NOT NULL CHECK(preference IN ('prefer','avoid','neutral')),
                updated_at TEXT NOT NULL,
                PRIMARY KEY(user_id, activity)
            );
            CREATE INDEX IF NOT EXISTS observations_interval ON observations(session_id, t_start, t_end);
            CREATE INDEX IF NOT EXISTS session_user ON sessions(user_id);
            CREATE TABLE IF NOT EXISTS observation_feedback (
                feedback_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                event_id TEXT, reference_candidate INTEGER NOT NULL,
                created_at TEXT NOT NULL, payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS feedback_session ON observation_feedback(session_id, reference_candidate);
            CREATE TABLE IF NOT EXISTS trial_enrollments (
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                mode TEXT NOT NULL, sequence TEXT, allocated INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(user_id,mode)
            );
            CREATE TABLE IF NOT EXISTS trial_periods (
                session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
                period INTEGER NOT NULL CHECK(period IN (1,2)), active_seconds REAL
            );
            CREATE TABLE IF NOT EXISTS trial_ratings (
                session_id TEXT NOT NULL REFERENCES trial_periods(session_id) ON DELETE CASCADE,
                decision_id TEXT NOT NULL, wanted INTEGER NOT NULL CHECK(wanted IN (0,1)),
                PRIMARY KEY(session_id,decision_id)
            );
        """)
        # Enforce the same opt-in inside SQLite as well as in the Python writer.
        # This guards future code paths that accidentally bypass _safe_payload.
        for table in ("observations", "utterances", "interventions", "events"):
            for operation in ("INSERT", "UPDATE"):
                self._db.execute(f"""
                    CREATE TRIGGER IF NOT EXISTS privacy_{table}_{operation.lower()}
                    BEFORE {operation} ON {table}
                    WHEN COALESCE((SELECT
                        CASE WHEN json_type(consent_json,'$.save_transcript') IS NOT NULL
                             THEN json_extract(consent_json,'$.save_transcript')
                             ELSE json_extract(consent_json,'$.transcript') END
                        FROM sessions WHERE id=NEW.session_id),0) != 1
                    AND EXISTS (SELECT 1 FROM json_tree(NEW.payload_json)
                        WHERE lower(CAST(key AS TEXT)) IN
                        ('text','transcript','transcription','raw_text','asr_text',
                         'transcript_text','utterance_text','recognized_text'))
                    BEGIN SELECT RAISE(ABORT,'transcript persistence requires consent'); END
                """)
        self._db.commit()
        self._revocation_barrier = set()
        for session_id in self.pending_transcript_cleanup():
            try:
                self.set_transcript_consent(session_id, False)
            except sqlite3.Error:
                # The journal still denies new writes and masks derived data on export.
                continue

    def pending_transcript_cleanup(self):
        with self._lock:
            rows = self._db.execute("SELECT session_id FROM pending_transcript_cleanup")
            return sorted(self._revocation_barrier | {row[0] for row in rows})

    def request_transcript_cleanup(self, session_id):
        """Stop newly queued writes immediately and journal the intent before async cleanup."""
        with self._lock, self._db:
            self._revocation_barrier.add(session_id)
            self._db.execute("INSERT OR IGNORE INTO pending_transcript_cleanup VALUES (?)", (session_id,))

    def _transcript_cleanup_requested(self, session_id):
        return (session_id in self._revocation_barrier or self._db.execute(
            "SELECT 1 FROM pending_transcript_cleanup WHERE session_id=?", (session_id,)).fetchone() is not None)

    @staticmethod
    def _json(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)

    def _session(self, session_id):
        row = self._db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise ValueError("session does not exist")
        return row

    def _safe_payload(self, session_id, payload):
        consent = json.loads(self._session(session_id)["consent_json"])
        # Accepted canonical opt-in key: save_transcript. Compatibility with
        # transcript is explicit; missing permission always means no transcript.
        permitted = consent.get("save_transcript", consent.get("transcript", False)) is True
        if permitted and not self._transcript_cleanup_requested(session_id):
            return deepcopy(payload)
        private_keys = {"text", "transcript", "transcription", "raw_text", "asr_text", "semantic_context", "evidence_text",
                        "transcript_text", "utterance_text", "recognized_text"}
        def scrub(value):
            if isinstance(value, dict):
                return {key: scrub(item) for key, item in value.items()
                        if str(key).lower() not in private_keys}
            if isinstance(value, (list, tuple)):
                return [scrub(item) for item in value]
            return deepcopy(value)
        return scrub(payload)

    def create_user(self, name) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("profile name cannot be empty")
        user_id = str(uuid.uuid4())
        with self._lock, self._db:
            self._db.execute("INSERT INTO users VALUES (?, ?, ?)", (user_id, name.strip(), utc_now()))
        return user_id

    def list_users(self) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self._db.execute("SELECT id, name FROM users ORDER BY created_at, id")]

    def start_session(self, user_id, consent: dict, mode: str) -> str:
        if not isinstance(consent, dict):
            raise ValueError("consent must be a dictionary")
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError("mode must be a nonempty string")
        session_id = str(uuid.uuid4())
        with self._lock, self._db:
            if self._db.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone() is None:
                raise ValueError("profile does not exist")
            flags = dict(consent)
            trial = flags.pop("crossover_trial", False) is True
            self._db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, NULL, ?, ?)",
                (session_id, user_id, utc_now(), self._json(flags), mode),
            )
            if trial:
                if flags.get("collection") is not True or mode not in ("real", "simulation"):
                    raise ValueError("实验需本人明确同意采集并选择真实或模拟来源")
                entry = self._db.execute(
                    "SELECT sequence,allocated FROM trial_enrollments WHERE user_id=? AND mode=?",
                    (user_id, mode),
                ).fetchone()
                if entry and (entry["sequence"] is None or entry["allocated"] >= 2):
                    raise ValueError("该档案在此来源已退出或已分配两场实验；请取消实验勾选开始普通会话")
                if not entry:
                    self._db.execute(
                        "INSERT INTO trial_enrollments VALUES (?,?,?,0)",
                        (user_id, mode, secrets.choice(("AB", "BA"))),
                    )
                period = entry["allocated"] + 1 if entry else 1
                self._db.execute("INSERT INTO trial_periods VALUES (?,?,NULL)", (session_id, period))
                self._db.execute(
                    "UPDATE trial_enrollments SET allocated=? WHERE user_id=? AND mode=?",
                    (period, user_id, mode),
                )
        return session_id

    def trial_assignment(self, session_id):
        with self._lock:
            row = self._db.execute(
                """SELECT p.period,e.sequence,p.active_seconds FROM trial_periods p
                JOIN sessions s ON s.id=p.session_id JOIN trial_enrollments e USING(user_id,mode)
                WHERE p.session_id=?""",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "sequence": row["sequence"],
                "period": row["period"],
                "condition": arm_for(row["sequence"], row["period"]),
                "status": "assigned" if row["active_seconds"] is None else "complete",
            }

    def trial_status(self, user_id, mode):
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM trial_enrollments WHERE user_id=? AND mode=?", (user_id, mode)
            ).fetchone()
            if not row:
                return {"status": "new", "next_period": 1}
            if row["sequence"] is None:
                return {"status": "withdrawn", "next_period": None}
            n = row["allocated"]
            finished = self._db.execute(
                """SELECT count(*) FROM trial_periods p JOIN sessions s ON s.id=p.session_id
                WHERE s.user_id=? AND s.mode=? AND p.active_seconds IS NOT NULL""",
                (user_id, mode),
            ).fetchone()[0]
            return {
                "status": ("next" if n == 1 else "complete" if finished == 2 else "incomplete"),
                "next_period": 2 if n == 1 else None,
                "allocated": n,
                "completed": finished,
                "incomplete_periods": n - finished,
                "sequence": row["sequence"],
            }

    def withdraw_trial(self, user_id, mode):
        with self._lock, self._db:
            updated = self._db.execute(
                """UPDATE trial_enrollments SET sequence=NULL, allocated=0
                WHERE user_id=? AND mode=? AND sequence IS NOT NULL""",
                (user_id, mode),
            ).rowcount
            self._db.execute(
                """DELETE FROM trial_periods WHERE session_id IN
                (SELECT id FROM sessions WHERE user_id=? AND mode=?)""",
                (user_id, mode),
            )
            return bool(updated)

    def rate_trial_prompt(self, session_id, decision_id, wanted):
        with self._lock, self._db:
            if not decision_id or type(wanted) is not bool:
                return False
            return bool(
                self._db.execute(
                    """INSERT OR IGNORE INTO trial_ratings
                SELECT session_id,?,? FROM trial_periods WHERE session_id=? AND active_seconds IS NULL
                AND EXISTS(SELECT 1 FROM events WHERE session_id=? AND kind='prompt_asked'
                    AND json_extract(payload_json,'$.decision_id')=?)""",
                    (decision_id, int(wanted), session_id, session_id, decision_id),
                ).rowcount
            )

    def trial_periods(self, mode):
        # Only the fields used by the statistical protocol; never read text, names, or raw observations.
        with self._lock:
            rows = self._db.execute(
                """SELECT p.*,s.mode,s.user_id,e.sequence FROM trial_periods p
                JOIN sessions s ON p.session_id=s.id JOIN trial_enrollments e USING(user_id,mode)
                WHERE s.mode=? AND e.sequence IS NOT NULL ORDER BY s.user_id,p.period""",
                (mode,),
            ).fetchall()
            result = []
            for row in rows:
                p = dict(row)
                sid = p.pop("session_id")
                p["condition"] = arm_for(p["sequence"], p["period"])
                p["events"] = [
                    (
                        e[0],
                        {
                            "decision_id": e[1],
                            "accepted": bool(e[2]) if e[2] is not None else None,
                            "ranking_changed": bool(e[3]),
                        },
                    )
                    for e in self._db.execute(
                        """SELECT kind,json_extract(payload_json,'$.decision_id'),
                    json_extract(payload_json,'$.accepted'),
                    json_extract(payload_json,'$.shadow.ranking') != COALESCE(json_extract(payload_json,'$.baseline_ranking'),
                    json_extract(payload_json,'$.ranking')) FROM events WHERE session_id=? AND kind IN
                    ('prompt_asked','prompt_decision','intervention_confirmation','recommendation_offered') ORDER BY id""",
                        (sid,),
                    )
                ]
                intervals = [
                    (r[0], r[1])
                    for r in self._db.execute(
                        """SELECT t_start,t_end FROM observations
                    WHERE session_id=? AND modality IN ('visual','audio') AND valid=1
                    AND json_extract(payload_json,'$.quality')>=0.2""",
                        (sid,),
                    )
                ]
                p["observed_seconds"] = interval_duration(intervals)
                p["activities"] = [
                    dict(a)
                    for a in self._db.execute(
                        """SELECT
                    json_extract(payload_json,'$.activity') AS activity, json_extract(payload_json,'$.status') AS status,
                    json_extract(payload_json,'$.pre') AS pre,json_extract(payload_json,'$.post') AS post,
                    json_extract(payload_json,'$.followup.status') AS followup_status,
                    json_extract(payload_json,'$.followup.distress') AS followup_distress
                    FROM interventions WHERE session_id=?""",
                        (sid,),
                    )
                ]
                p["ratings"] = {
                    r[0]: bool(r[1])
                    for r in self._db.execute(
                        "SELECT decision_id,wanted FROM trial_ratings WHERE session_id=?", (sid,)
                    )
                }
                result.append(p)
            return result

    def trial_flow(self, mode):
        with self._lock:
            row = self._db.execute(
                """SELECT COUNT(*) AS randomized, COUNT(*) FILTER(WHERE sequence IS NULL)
                AS withdrawn,COALESCE(SUM(allocated),0) AS periods_allocated FROM trial_enrollments WHERE mode=?""",
                (mode,),
            ).fetchone()
            return dict(row)

    def analyze_trial(self, mode):
        from .crossover import analyze

        # One coherent snapshot, including paired outcome/withdrawal and followup revisions.
        with self._lock:
            self._db.execute("BEGIN")
            try:
                periods, flow = self.trial_periods(mode), self.trial_flow(mode)
            finally:
                self._db.rollback()
        return analyze(periods, mode, flow)

    @classmethod
    def read_trial_report(cls, database, mode):
        """Open an existing study database read-only (never create tables or run migrations)."""
        uri = Path(database).resolve().as_uri() + "?mode=ro"
        reader = cls.__new__(cls)
        reader._lock = threading.RLock()
        reader._db = sqlite3.connect(uri, uri=True)
        reader._db.row_factory = sqlite3.Row
        try:
            return reader.analyze_trial(mode)
        finally:
            reader._db.close()

    def save_observation(self, session_id, observation):
        event_id = observation.get("event_id")
        if not event_id:
            raise ValueError("observation needs an event_id")
        with self._lock, self._db:
            payload = self._safe_payload(session_id, observation)
            if payload.get("session_id", session_id) != session_id:
                raise ValueError("observation belongs to another session")
            payload["session_id"] = session_id
            self._db.execute("""INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, event_id) DO UPDATE SET
                modality=excluded.modality,t_start=excluded.t_start,t_end=excluded.t_end,
                valid=excluded.valid,payload_json=excluded.payload_json""",
                             (session_id, event_id, payload["modality"], payload["t_start"],
                              payload["t_end"], bool(payload.get("valid")), self._json(payload)))

    def save_utterance(self, session_id, utterance: dict, require_consent=False):
        utterance_id = utterance.get("id") or utterance.get("event_id") or str(uuid.uuid4())
        with self._lock, self._db:
            consent = json.loads(self._session(session_id)["consent_json"])
            if self._transcript_cleanup_requested(session_id):
                return None
            permitted = consent.get("save_transcript", consent.get("transcript", False)) is True
            if not permitted and (require_consent or consent.get("transcript_revoked_at")):
                return None
            payload = self._safe_payload(session_id, utterance)
            if payload.get("session_id", session_id) != session_id:
                raise ValueError("utterance belongs to another session")
            payload["session_id"] = session_id
            payload["id"] = utterance_id
            self._db.execute("""INSERT INTO utterances VALUES (?, ?, ?)
                ON CONFLICT(session_id, id) DO UPDATE SET payload_json=excluded.payload_json""",
                             (session_id, utterance_id, self._json(payload)))
        return utterance_id

    def set_transcript_consent(self, session_id, allowed):
        """Change future ASR persistence permission without erasing prior history."""
        if not isinstance(allowed, bool):
            raise ValueError("transcript permission must be a boolean")
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            consent = json.loads(self._session(session_id)["consent_json"])
            previous = consent.get("save_transcript", consent.get("transcript", False)) is True
            pending = self._transcript_cleanup_requested(session_id)
            if allowed and pending:
                raise ValueError("pending transcript cleanup")
            if previous == allowed and consent.get("transcript") is allowed and consent.get("save_transcript") is allowed:
                if pending:
                    self._db.execute("DELETE FROM pending_transcript_cleanup WHERE session_id=?", (session_id,))
                    self._revocation_barrier.discard(session_id)
                return consent
            now = utc_now()
            revision = consent.get("transcript_consent_revision", 0)
            revision = revision if isinstance(revision, int) and not isinstance(revision, bool) else 0
            consent.update(transcript=allowed, save_transcript=allowed,
                           transcript_consent_revision=revision + 1,
                           transcript_consent_updated_at=now)
            if not allowed:
                consent["transcript_revoked_at"] = now
                self._db.execute("""UPDATE utterances SET payload_json=json_remove(payload_json,'$.semantic_context')
                    WHERE session_id=? AND json_type(payload_json,'$.semantic_context') IS NOT NULL""", (session_id,))
                self._db.execute("DELETE FROM pending_transcript_cleanup WHERE session_id=?", (session_id,))
            self._db.execute("UPDATE sessions SET consent_json=? WHERE id=?", (self._json(consent), session_id))
            self._db.execute("INSERT INTO events(session_id,kind,created_at,payload_json) VALUES (?, ?, ?, ?)",
                (session_id, "transcript_consent_changed", now,
                 self._json({"allowed": allowed, "previously_allowed": previous,
                             "retains_authorized_history": True, "revision": revision + 1})))
        with self._lock:
            if not allowed:
                self._revocation_barrier.discard(session_id)
        return deepcopy(consent)

    def save_self_report(self, session_id, phase: str, distress: int, intervention_id=None) -> str:
        if isinstance(distress, bool) or not isinstance(distress, int) or not 0 <= distress <= 10:
            raise ValueError("distress must be an integer from 0 to 10")
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError("phase cannot be empty")
        report_id = str(uuid.uuid4())
        with self._lock, self._db:
            self._session(session_id)
            self._db.execute("INSERT INTO self_reports VALUES (?, ?, ?, ?, ?, ?)",
                             (report_id, session_id, phase, distress, intervention_id, utc_now()))
        return report_id

    def save_observation_feedback(self, record: FeedbackRecord):
        with self._lock, self._db:
            session = self._session(record.session_id)
            if session["user_id"] != record.user_id or session["ended_at"]:
                raise ValueError("反馈所属的会话已结束或档案不一致")
            payload = self._safe_payload(record.session_id, record.payload())
            consent = json.loads(session["consent_json"])
            payload["learning_consent"] = record.learning_consent and consent.get("personal_learning") is True
            evidence = payload["evidence"]
            eligible = bool(payload["learning_consent"] and record.feedback_type == "typical_reference"
                            and evidence.get("fresh_at_feedback") and comparison_key(evidence))
            stamp = utc_now()
            payload.update(mode=session["mode"], created_at=stamp,
                           reference_candidate_at_feedback=eligible)
            if evidence.get("event_id") and record.feedback_type in ("subjective_correction", "distress_report"):
                self._db.execute("""UPDATE observation_feedback SET reference_candidate=0
                    WHERE session_id=? AND event_id=?""", (record.session_id, evidence["event_id"]))
            self._db.execute("INSERT INTO observation_feedback VALUES (?, ?, ?, ?, ?, ?)",
                (record.feedback_id, record.session_id, evidence.get("event_id"), eligible, stamp, self._json(payload)))
            self._db.execute("INSERT INTO events(session_id,kind,created_at,payload_json) VALUES (?, ?, ?, ?)",
                (record.session_id, "user_correction" if record.feedback_type == "subjective_correction"
                 else "observation_feedback", stamp, self._json({"feedback_id": record.feedback_id,
                 "feedback_type": record.feedback_type, "label": record.user_label,
                 "target_event_ids": record.target_event_ids, "learning_consent": payload["learning_consent"]})))
            return payload

    def pause_personal_learning(self, session_id):
        with self._lock, self._db:
            session = self._session(session_id)
            consent = json.loads(session["consent_json"])
            consent["personal_learning"] = False
            self._db.execute("UPDATE sessions SET consent_json=? WHERE id=?", (self._json(consent), session_id))

    def reset_personal_reference(self, user_id, mode):
        """Exclude all prior reference candidates without rewriting the original feedback."""
        with self._lock, self._db:
            self._db.execute("""UPDATE observation_feedback SET reference_candidate=0
                WHERE reference_candidate=1 AND session_id IN
                (SELECT id FROM sessions WHERE user_id=? AND mode=?)""", (user_id, mode))
            self._db.execute("""UPDATE sessions SET consent_json=json_set(consent_json, '$.personal_learning', json('false'))
                WHERE user_id=? AND mode=? AND ended_at IS NULL""", (user_id, mode))

    def personal_reference(self, user_id, mode):
        """Use opt-in references from completed sessions and verify the final evidence revision."""
        if mode not in ("real", "simulation"):
            return build_reference([])
        with self._lock:
            rows = self._db.execute("""SELECT f.payload_json, o.payload_json AS final_observation
                FROM sessions s JOIN observation_feedback f ON f.session_id=s.id
                JOIN observations o ON o.session_id=s.id AND o.event_id=f.event_id
                WHERE s.user_id=? AND s.mode=? AND s.ended_at IS NOT NULL AND f.reference_candidate=1
                AND NOT EXISTS (SELECT 1 FROM json_each(f.payload_json,'$.evidence.source_ids') src
                    WHERE NOT EXISTS (SELECT 1 FROM observations raw WHERE raw.session_id=f.session_id
                        AND raw.event_id=src.value AND raw.valid=1 AND raw.modality IN ('visual','audio')))
                ORDER BY f.created_at DESC, f.rowid DESC LIMIT 512""", (user_id, mode))
            records = []
            for row in rows:
                record = json.loads(row["payload_json"])
                final = json.loads(row["final_observation"])
                if matches_final_revision(record, final):
                    records.append(record)
            return build_reference(records)

    def save_intervention(self, session_id, record: dict) -> str:
        record_id = record.get("id") or str(uuid.uuid4())
        with self._lock, self._db:
            payload = self._safe_payload(session_id, record)
            payload["id"] = record_id
            self._db.execute("""INSERT INTO interventions VALUES (?, ?, ?)
                ON CONFLICT(session_id, id) DO UPDATE SET payload_json=excluded.payload_json""",
                             (session_id, record_id, self._json(payload)))
        return record_id

    def _save_report_in_transaction(self, session_id, intervention_id, phase, distress):
        """Caller owns the lock/transaction; never commit a partial bundle."""
        if isinstance(distress, bool) or not isinstance(distress, int) or not 0 <= distress <= 10:
            raise ValueError("distress must be an integer from 0 to 10")
        if phase not in ("pre", "post", "followup"):
            raise ValueError("unknown assessment phase")
        old = self._db.execute("""SELECT id FROM self_reports
            WHERE session_id=? AND intervention_id=? AND phase=? ORDER BY created_at,id LIMIT 1""",
            (session_id, intervention_id, phase)).fetchone()
        report_id = old["id"] if old else uuid.uuid5(uuid.NAMESPACE_URL,
            f"emotion-assistant:{session_id}:{intervention_id}:{phase}").hex
        self._db.execute("""INSERT INTO self_reports VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET distress=excluded.distress""",
            (report_id, session_id, phase, distress, intervention_id, utc_now()))
        return report_id

    def save_activity_bundle(self, session_id, record, reports=None, preference="unchanged",
                             event_kind=None, event_payload=None):
        """Atomically save an activity, its ratings, explicit preference and event.

        Existing JSON fields and privacy enforcement are preserved. Public single
        record methods remain available, but must not be nested here because
        their connection context managers would commit this transaction early.
        """
        from .regulation import ACTIVITIES
        if not isinstance(record, dict):
            raise ValueError("activity record must be a dictionary")
        if reports is not None and not isinstance(reports, (list, tuple)):
            raise ValueError("reports must be a list")
        if preference not in ("unchanged", "neutral", "prefer", "avoid"):
            raise ValueError("unknown preference")
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            session = self._session(session_id)
            record_id = record.get("id") or str(uuid.uuid4())
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("activity needs a string id")
            identities = {"id": record_id, "session_id": session_id,
                          "user_id": session["user_id"], "mode": session["mode"]}
            if any(key in record and record[key] != value for key, value in identities.items()):
                raise ValueError("activity identity cannot be changed")
            payload = self._safe_payload(session_id, record)
            payload.update(identities)
            payload["updated_at"] = utc_now()
            self._db.execute("""INSERT INTO interventions VALUES (?, ?, ?)
                ON CONFLICT(session_id,id) DO UPDATE SET payload_json=excluded.payload_json""",
                (session_id, record_id, self._json(payload)))
            for report in reports or []:
                if not isinstance(report, dict):
                    raise ValueError("each report must be a dictionary")
                self._save_report_in_transaction(session_id, record_id,
                                                 report.get("phase"), report.get("distress"))
            if preference != "unchanged":
                activity = payload.get("activity")
                if activity not in ACTIVITIES:
                    raise ValueError("unknown activity")
                self._db.execute("""INSERT INTO activity_preferences VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id,activity) DO UPDATE SET
                    preference=excluded.preference,updated_at=excluded.updated_at""",
                    (session["user_id"], activity, preference, utc_now()))
            if event_kind is not None:
                if not isinstance(event_kind, str) or not event_kind:
                    raise ValueError("event kind must be a nonempty string")
                event = deepcopy(event_payload) if event_payload is not None else {}
                if not isinstance(event, dict):
                    raise ValueError("event payload must be a dictionary")
                event["intervention_id"] = record_id
                safe_event = self._safe_payload(session_id, event)
                self._db.execute("INSERT INTO events(session_id,kind,created_at,payload_json) VALUES (?, ?, ?, ?)",
                    (session_id, event_kind, utc_now(), self._json(safe_event)))
            return deepcopy(payload)

    def complete_followup(self, session_id, intervention_id, user_id, mode, distress,
                          helpfulness="unknown"):
        """Resolve a pending follow-up once with all related rows in one commit."""
        from .regulation import _helpfulness, _mode, _rating, evaluate_outcome
        if distress is not None and _rating(distress) is None:
            raise ValueError("distress must be an integer from 0 to 10 or None")
        helpfulness = _helpfulness(helpfulness)
        if helpfulness not in ("helpful", "no_change", "worse", "unknown"):
            raise ValueError("unknown helpfulness")
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            session = self._session(session_id)
            if (session["user_id"] != user_id or _mode(mode) is None
                    or _mode(session["mode"]) != _mode(mode)):
                raise ValueError("follow-up belongs to another profile or mode")
            row = self._db.execute("SELECT payload_json FROM interventions WHERE session_id=? AND id=?",
                                   (session_id, intervention_id)).fetchone()
            if row is None:
                raise ValueError("intervention does not exist")
            payload = json.loads(row["payload_json"])
            followup = payload.get("followup")
            if (payload.get("status") != "completed" or not isinstance(followup, dict)
                    or followup.get("status") != "pending"):
                return None
            now = datetime.now(timezone.utc)
            due = self._as_utc(followup["due_at"])
            pre, post = _rating(payload.get("pre")), _rating(payload.get("post"))
            outcome = evaluate_outcome(pre, distress, helpfulness)
            followup.update(status="completed" if distress is not None else "skipped",
                completed_at=now.isoformat(timespec="milliseconds"), distress=distress,
                helpfulness=helpfulness, answered_early=now < due, loop_result=outcome,
                delta=pre - distress if pre is not None and distress is not None else None,
                delta_from_immediate=post - distress if post is not None and distress is not None else None)
            payload.update(followup=followup, updated_at=utc_now(),
                discomfort=bool(payload.get("discomfort") or helpfulness == "worse"))
            if payload["discomfort"]:
                payload.setdefault("numeric_evaluation", payload.get("evaluation"))
                payload["evaluation"] = "discomfort"
            payload.update(session_id=session_id, user_id=session["user_id"], mode=session["mode"])
            safe = self._safe_payload(session_id, payload)
            self._db.execute("UPDATE interventions SET payload_json=? WHERE session_id=? AND id=?",
                             (self._json(safe), session_id, intervention_id))
            if distress is not None:
                self._save_report_in_transaction(session_id, intervention_id, "followup", distress)
            event = {"intervention_id": intervention_id, "status": followup["status"],
                     "answered_early": followup["answered_early"], "loop_result": outcome}
            self._db.execute("INSERT INTO events(session_id,kind,created_at,payload_json) VALUES (?, ?, ?, ?)",
                             (session_id, "followup_completed", utc_now(), self._json(event)))
            return deepcopy(safe)

    @staticmethod
    def _intervention_payload(row):
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        # Identity/mode is authoritative session metadata, never caller JSON.
        payload.update(session_id=row["session_id"], user_id=row["user_id"], mode=row["mode"])
        return payload

    def list_interventions(self, user_id=None, mode=None, limit=200) -> list[dict]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5000:
            raise ValueError("limit must be an integer between 1 and 5000")
        query = """SELECT i.session_id,i.payload_json,s.user_id,s.mode FROM interventions i
            JOIN sessions s ON s.id=i.session_id WHERE 1=1"""
        parameters = []
        if user_id is not None:
            query += " AND s.user_id=?"
            parameters.append(user_id)
        if mode is not None:
            if mode == "simulation":
                query += " AND s.mode IN ('simulation','demo','replay')"
            else:
                query += " AND s.mode=?"
                parameters.append(mode)
        query += " ORDER BY COALESCE(json_extract(i.payload_json,'$.ended_at'),json_extract(i.payload_json,'$.started_at'),s.started_at) DESC,i.rowid DESC LIMIT ?"
        parameters.append(limit)
        with self._lock:
            return [self._intervention_payload(row) for row in self._db.execute(query, parameters)]

    def historical_discomfort(self, user_id, mode):
        """Rare full-history safety check when the ranking sample has reached its cap."""
        from .regulation import _discomfort
        modes = ("simulation", "demo", "replay") if mode == "simulation" else ("real",)
        marks = ",".join("?" for _ in modes)
        with self._lock:
            rows = self._db.execute(f"""SELECT i.payload_json FROM interventions i
                JOIN sessions s ON s.id=i.session_id WHERE s.user_id=? AND s.mode IN ({marks})""",
                (user_id, *modes))
            return {entry.get("activity") for row in rows
                    if _discomfort(entry := json.loads(row[0]))}

    def recent_prompt_history(self, user_id, mode, since):
        with self._lock:
            rows = self._db.execute("""SELECT e.kind, e.created_at FROM events e
                JOIN sessions s ON e.session_id=s.id WHERE s.user_id=? AND s.mode=?
                AND e.kind IN ('prompt_asked','prompt_declined') AND e.created_at>=?
                ORDER BY e.created_at DESC,e.id DESC LIMIT 128""", (user_id, mode, since)).fetchall()
        return [(row["kind"].removeprefix("prompt_"), datetime.fromisoformat(row["created_at"]).timestamp())
                for row in reversed(rows)]

    def get_intervention(self, session_id, intervention_id) -> dict | None:
        with self._lock:
            row = self._db.execute("""SELECT i.session_id,i.payload_json,s.user_id,s.mode
                FROM interventions i JOIN sessions s ON s.id=i.session_id
                WHERE i.session_id=? AND i.id=?""", (session_id, intervention_id)).fetchone()
            return self._intervention_payload(row)

    def patch_intervention(self, session_id, intervention_id, updates: dict) -> dict:
        if not isinstance(updates, dict):
            raise ValueError("updates must be a dictionary")
        with self._lock, self._db:
            session = self._session(session_id)
            row = self._db.execute("SELECT payload_json FROM interventions WHERE session_id=? AND id=?",
                                   (session_id, intervention_id)).fetchone()
            if row is None:
                raise ValueError("intervention does not exist")
            identities = {"id": intervention_id, "session_id": session_id,
                          "user_id": session["user_id"], "mode": session["mode"]}
            if any(key in updates and updates[key] != value for key, value in identities.items()):
                raise ValueError("intervention identity cannot be changed")
            payload = json.loads(row["payload_json"])
            payload.update(deepcopy(updates))
            payload["id"] = intervention_id
            payload["updated_at"] = utc_now()
            safe = self._safe_payload(session_id, payload)
            self._db.execute("UPDATE interventions SET payload_json=? WHERE session_id=? AND id=?",
                             (self._json(safe), session_id, intervention_id))
            safe.update(session_id=session_id, user_id=session["user_id"], mode=session["mode"])
            return safe

    def save_activity_preference(self, user_id, activity, preference):
        from .regulation import ACTIVITIES
        if activity not in ACTIVITIES or preference not in ("prefer", "avoid", "neutral"):
            raise ValueError("unknown activity or preference")
        with self._lock, self._db:
            if self._db.execute("SELECT 1 FROM users WHERE id=?", (user_id,)).fetchone() is None:
                raise ValueError("profile does not exist")
            self._db.execute("""INSERT INTO activity_preferences VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id,activity) DO UPDATE SET preference=excluded.preference,
                updated_at=excluded.updated_at""", (user_id, activity, preference, utc_now()))

    def get_activity_preferences(self, user_id) -> dict:
        with self._lock:
            return {row["activity"]: row["preference"] for row in self._db.execute(
                "SELECT activity,preference FROM activity_preferences WHERE user_id=? ORDER BY activity", (user_id,))}

    @staticmethod
    def _as_utc(value):
        if isinstance(value, datetime):
            stamp = value
        else:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(timezone.utc)

    def list_pending_followups(self, user_id, mode=None, now=None, due_only=False) -> list[dict]:
        current = self._as_utc(now or utc_now())
        result = []
        # The bounded query supplies authoritative identity and mode metadata.
        for record in self.list_interventions(user_id=user_id, mode=mode, limit=5000):
            followup = record.get("followup")
            if (record.get("status") != "completed" or not isinstance(followup, dict)
                    or followup.get("status") != "pending"):
                continue
            try:
                due_at = self._as_utc(followup["due_at"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            due = due_at <= current
            if due_only and not due:
                continue
            record["due"] = due
            record["remaining_seconds"] = max(0, math.ceil((due_at - current).total_seconds()))
            result.append(record)
        result.sort(key=lambda record: self._as_utc(record["followup"]["due_at"]))
        return result

    def append_event(self, session_id, kind, payload: dict):
        with self._lock, self._db:
            safe = self._safe_payload(session_id, payload)
            self._db.execute("INSERT INTO events(session_id,kind,created_at,payload_json) VALUES (?, ?, ?, ?)",
                             (session_id, kind, utc_now(), self._json(safe)))

    def finish_session(self, session_id, trial_active_seconds=None):
        with self._lock, self._db:
            self._session(session_id)
            self._db.execute("UPDATE sessions SET ended_at=COALESCE(ended_at, ?) WHERE id = ?",
                             (utc_now(), session_id))
            if trial_active_seconds is not None:
                self._db.execute("UPDATE trial_periods SET active_seconds=COALESCE(active_seconds,?) WHERE session_id=?",
                                 (max(0.,float(trial_active_seconds)),session_id))

    def list_sessions(self, user_id=None) -> list[dict]:
        from .regulation import _discomfort
        with self._lock:
            sql = """SELECT s.*, u.name AS user_name,
                (SELECT count(*) FROM observations o WHERE o.session_id=s.id) AS observation_count,
                (SELECT count(*) FROM observations o WHERE o.session_id=s.id
                    AND o.modality IN ('visual','audio')) AS raw_observation_count,
                (SELECT count(*) FROM interventions i WHERE i.session_id=s.id) AS intervention_count,
                (SELECT payload_json FROM interventions i WHERE i.session_id=s.id
                    ORDER BY COALESCE(json_extract(payload_json,'$.ended_at'),
                        json_extract(payload_json,'$.started_at'),'') DESC, rowid DESC
                    LIMIT 1) AS latest_intervention_json
                FROM sessions s JOIN users u ON u.id=s.user_id"""
            parameters = (user_id,) if user_id is not None else ()
            rows = self._db.execute(sql + (" WHERE s.user_id=?" if user_id is not None else "")
                                    + " ORDER BY s.started_at DESC", parameters)
            result = []
            for row in rows:
                item = dict(row)
                item["consent"] = json.loads(item.pop("consent_json"))
                latest = json.loads(item.pop("latest_intervention_json") or "null")
                item["last_evaluation"] = latest.get("evaluation") if latest else None
                item["last_numeric_evaluation"] = latest.get("numeric_evaluation", latest.get("evaluation")) if latest else None
                item["last_delta"] = latest.get("delta") if latest else None
                item["last_helpfulness"] = latest.get("helpfulness") if latest else None
                # Keep numeric pre/post fields intact. The displayed feedback
                # can additionally prioritize explicit process/delayed discomfort.
                loop = latest.get("loop_result") if latest else None
                loop = loop if isinstance(loop, dict) else {}
                followup = latest.get("followup") if latest else None
                followup = followup if isinstance(followup, dict) else {}
                delayed_loop = followup.get("loop_result")
                if followup.get("status") == "completed" and isinstance(delayed_loop, dict):
                    loop = delayed_loop
                discomfort = _discomfort(latest or {})
                item["latest_discomfort"] = discomfort
                item["latest_loop_state"] = "discomfort" if discomfort else loop.get("state")
                item["latest_loop_caption"] = (
                    loop.get("caption") if not discomfort or loop.get("state") == "discomfort"
                    else "你报告不舒服；这条反馈优先于数值下降。")
                item["latest_followup_status"] = followup.get("status")
                item["duration_seconds"] = max(0.0, (
                    datetime.fromisoformat(item["ended_at"] or utc_now())
                    - datetime.fromisoformat(item["started_at"])).total_seconds())
                result.append(item)
            # Read only compact interval columns, never entire transcript/model
            # payloads, when building the history page's deduplicated coverage.
            by_session = {item["id"]: {"visual": [], "audio": [], "joint": []} for item in result}
            range_sql = """SELECT o.session_id,o.modality,o.t_start,o.t_end
                FROM observations o JOIN sessions s ON s.id=o.session_id
                WHERE o.valid=1 AND o.modality IN ('visual','audio')
                AND json_type(o.payload_json,'$.quality') IN ('integer','real')
                AND json_extract(o.payload_json,'$.quality') >= 0.2"""
            for row in self._db.execute(range_sql + (" AND s.user_id=?" if user_id is not None else ""), parameters):
                by_session[row["session_id"]][row["modality"]].append((row["t_start"], row["t_end"]))
            for item in result:
                item["coverage"] = self._coverage(by_session[item["id"]])
            return result

    @staticmethod
    def _coverage(intervals):
        intervals = {key: list(value) for key, value in intervals.items()}
        for a, b in interval_union(intervals["visual"]):
            for c, d in interval_union(intervals["audio"]):
                if min(b, d) > max(a, c):
                    intervals["joint"].append((max(a, c), min(b, d)))
        coverage = {key + "_seconds": interval_duration(value) for key, value in intervals.items()}
        coverage["intervals"] = {key: interval_union(value) for key, value in intervals.items()}
        coverage["observed_seconds"] = interval_duration(intervals["visual"] + intervals["audio"])
        coverage["note"] = "Intervals are unioned; uncovered time is missing data, not neutral emotion."
        return coverage

    def export_session(self, session_id, path) -> Path:
        with self._lock:
            session = dict(self._session(session_id))
            session["consent"] = json.loads(session.pop("consent_json"))
            user = dict(self._db.execute("SELECT id,name FROM users WHERE id=?", (session["user_id"],)).fetchone())
            def payloads(table, order="rowid"):
                return [json.loads(row[0]) for row in self._db.execute(
                    f"SELECT payload_json FROM {table} WHERE session_id=? ORDER BY {order}", (session_id,))]
            observations = payloads("observations", "t_start,event_id")
            events = []
            for row in self._db.execute("SELECT id,kind,created_at,payload_json FROM events WHERE session_id=? ORDER BY id", (session_id,)):
                item = dict(row)
                item["payload"] = json.loads(item.pop("payload_json"))
                events.append(item)
            intervals = {key: [] for key in ("visual", "audio", "joint")}
            for observation in observations:
                if not observation.get("valid") or float(observation.get("quality", 0)) < 0.2:
                    continue
                modality = observation.get("modality")
                if modality in ("visual", "audio"):
                    intervals[modality].append((observation["t_start"], observation["t_end"]))
            # Intersection of actual usable visual/audio intervals is the joint
            # observation duration; a late inference arrival does not move it.
            coverage = self._coverage(intervals)
            coverage["sampling_span_end_seconds"] = max((float(item["t_end"]) for item in observations), default=0.0)
            coverage["session_elapsed_seconds"] = max(0.0, (
                datetime.fromisoformat(session["ended_at"] or utc_now())
                - datetime.fromisoformat(session["started_at"])).total_seconds())
            result = {"schema_version": 1, "exported_at": utc_now(), "user": user,
                      "session": session, "observations": observations, "utterances": payloads("utterances"),
                      "observation_feedback": [dict(json.loads(row["payload_json"]),
                          reference_candidate=bool(row["reference_candidate"])) for row in self._db.execute(
                          "SELECT payload_json,reference_candidate FROM observation_feedback WHERE session_id=? ORDER BY rowid",
                          (session_id,))],
                      "self_reports": [dict(row) for row in self._db.execute(
                          "SELECT * FROM self_reports WHERE session_id=? ORDER BY created_at,id", (session_id,))],
                      "interventions": payloads("interventions"), "events": events, "coverage": coverage,
                      "profile_preferences": self.get_activity_preferences(session["user_id"])}
            if self._transcript_cleanup_requested(session_id):
                session["consent"].update(transcript=False, save_transcript=False, semantic_cleanup_pending=True)
            if not session["consent"].get("save_transcript", session["consent"].get("transcript", False)):
                for utterance in result["utterances"]:
                    utterance.pop("semantic_context", None)
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            return destination

    def delete_session(self, session_id):
        with self._lock, self._db:
            self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._revocation_barrier.discard(session_id)

    def close(self):
        with self._lock:
            self._db.close()
