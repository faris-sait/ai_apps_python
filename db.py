"""Database helpers for lip-sync jobs/logs (SQLModel)

Environment variables:
- DATABASE_URL : Postgres URL like postgresql://user:pass@host:port/dbname
  REQUIRED - No fallback to SQLite. Must use Supabase Postgres.
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Optional, List

from sqlmodel import SQLModel, Field, create_engine, Session, select, Column, JSON

# Load environment variables from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

# Require DATABASE_URL - no fallback
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    # Try loading from parent directory .env as well
    try:
        from dotenv import load_dotenv
        import sys
        from pathlib import Path
        env_path = Path(__file__).parent / ".env"
        load_dotenv(env_path, override=False)
        DATABASE_URL = os.environ.get("DATABASE_URL")
    except:
        pass
    
    if not DATABASE_URL:
        raise ValueError(
            "DATABASE_URL environment variable is required. "
            "Please set it to your Supabase Postgres connection string: "
            "postgresql://postgres:password@db.xxx.supabase.co:5432/postgres"
        )

# If using Supabase, ensure SSL mode is required (many Supabase URLs need sslmode=require)
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
parsed = urlparse(DATABASE_URL)
# If URL contains un-encoded '@' in password (e.g. password with '@'), try to fix it by encoding the password
authority = parsed.netloc
if authority.count("@") > 1:
    # split at last '@' -> creds may contain ':' between user:password
    creds_part, host_part = authority.rsplit("@", 1)
    if ":" in creds_part:
        user, pwd = creds_part.split(":", 1)
        from urllib.parse import quote
        pwd_enc = quote(pwd, safe="")
        new_netloc = f"{user}:{pwd_enc}@{host_part}"
        parsed = parsed._replace(netloc=new_netloc)
        DATABASE_URL = urlunparse(parsed)

if parsed.scheme.startswith("postgres") and "supabase.co" in (parsed.hostname or ""):
    qs = parse_qs(parsed.query)
    if "sslmode" not in qs:
        qs["sslmode"] = ["require"]
        new_query = urlencode(qs, doseq=True)
        DATABASE_URL = urlunparse(parsed._replace(query=new_query))

# Ensure we're using Postgres/Supabase
if not parsed.scheme.startswith("postgres"):
    raise ValueError(
        f"Only PostgreSQL databases are supported. Got: {parsed.scheme}. "
        "Please use your Supabase connection string."
    )

engine = create_engine(DATABASE_URL, echo=False)



class JobModel(SQLModel, table=True):
    id: str = Field(primary_key=True, index=True)
    provider: str
    mode: str
    input_path: str
    audio_path: str
    output_path: Optional[str] = None
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    message: Optional[str] = None
    log_path: Optional[str] = None
    metadata_json: Optional[str] = None  # JSON stored as string


class JobLogModel(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="jobmodel.id")
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def create_job_record(job_dict: dict) -> None:
    with Session(engine) as session:
        model = JobModel(**job_dict)
        session.add(model)
        session.commit()


def update_job_record(job_id: str, updates: dict) -> None:
    with Session(engine) as session:
        stmt = select(JobModel).where(JobModel.id == job_id)
        res = session.exec(stmt).one_or_none()
        if not res:
            return
        for k, v in updates.items():
            setattr(res, k, v)
        session.add(res)
        session.commit()


def get_job_record(job_id: str) -> Optional[JobModel]:
    with Session(engine) as session:
        stmt = select(JobModel).where(JobModel.id == job_id)
        return session.exec(stmt).one_or_none()


def list_job_records() -> List[JobModel]:
    with Session(engine) as session:
        stmt = select(JobModel)
        return session.exec(stmt).all()


def append_job_log(job_id: str, content: str) -> None:
    with Session(engine) as session:
        log = JobLogModel(job_id=job_id, content=content)
        session.add(log)
        session.commit()


def get_job_logs(job_id: str) -> List[JobLogModel]:
    with Session(engine) as session:
        stmt = select(JobLogModel).where(JobLogModel.job_id == job_id).order_by(JobLogModel.created_at)
        return session.exec(stmt).all()


from sqlalchemy import text

def test_connection() -> bool:
    """Test DB connection; returns True if successful, False otherwise."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print("DB connection test failed:", e)
        return False