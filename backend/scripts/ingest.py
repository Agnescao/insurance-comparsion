from sqlalchemy.orm import Session

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
