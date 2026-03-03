import sys
from pathlib import Path

from sqlalchemy.orm import Session

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database import Base, engine
from app.services.ingestion import IngestionService


def main() -> None:
    Base.metadata.create_all(bind=engine)
    service = IngestionService()
    with Session(engine) as db:
        plans, chunks, facts = service.ingest_all(db)
        db.commit()
    print(f"ingested plans={plans}, chunks={chunks}, facts={facts}")


if __name__ == "__main__":
    main()
