"""Thread-safe local SQLite records with consent enforcement and JSON export."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
import json
from pathlib import Path
import sqlite3
import threading
import uuid

from .core import interval_duration, interval_union, utc_now


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
        if permitted:
            return deepcopy(payload)
        private_keys = {"text", "transcript", "transcription", "raw_text", "asr_text",
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
            self._db.execute("INSERT INTO sessions VALUES (?, ?, ?, NULL, ?, ?)",
                             (session_id, user_id, utc_now(), self._json(consent), mode))
        return session_id

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
            if previous == allowed and consent.get("transcript") is allowed and consent.get("save_transcript") is allowed:
                return consent
            now = utc_now()
            revision = consent.get("transcript_consent_revision", 0)
            revision = revision if isinstance(revision, int) and not isinstance(revision, bool) else 0
            consent.update(transcript=allowed, save_transcript=allowed,
                           transcript_consent_revision=revision + 1,
                           transcript_consent_updated_at=now)
            if not allowed:
                consent["transcript_revoked_at"] = now
            self._db.execute("UPDATE sessions SET consent_json=? WHERE id=?", (self._json(consent), session_id))
            self._db.execute("INSERT INTO events(session_id,kind,created_at,payload_json) VALUES (?, ?, ?, ?)",
                (session_id, "transcript_consent_changed", now,
                 self._json({"allowed": allowed, "previously_allowed": previous,
                             "retains_authorized_history": True, "revision": revision + 1})))
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

    def finish_session(self, session_id):
        with self._lock, self._db:
            self._session(session_id)
            self._db.execute("UPDATE sessions SET ended_at=COALESCE(ended_at, ?) WHERE id = ?",
                             (utc_now(), session_id))

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
                      "self_reports": [dict(row) for row in self._db.execute(
                          "SELECT * FROM self_reports WHERE session_id=? ORDER BY created_at,id", (session_id,))],
                      "interventions": payloads("interventions"), "events": events, "coverage": coverage,
                      "profile_preferences": self.get_activity_preferences(session["user_id"])}
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return destination

    def delete_session(self, session_id):
        with self._lock, self._db:
            self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def close(self):
        with self._lock:
            self._db.close()
