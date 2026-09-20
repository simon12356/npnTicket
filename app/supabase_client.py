import os
from typing import Generator
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL must be set in .env")

# psycopg2 does not support Prisma's 'pgbouncer' query parameter
if "pgbouncer" in DATABASE_URL:
    parsed = urlparse(DATABASE_URL)
    query_params = parse_qs(parsed.query)
    query_params.pop("pgbouncer", None)
    new_query = urlencode(query_params, doseq=True)
    DATABASE_URL = urlunparse(parsed._replace(query=new_query))

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=1800,
    connect_args={"sslmode": "require"},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()