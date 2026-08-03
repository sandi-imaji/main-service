"""
Database utilities and dependency injection
"""

from contextlib import contextmanager
from typing import Generator, Optional
from sqlmodel import Session
from app.database.db import engine


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
  """Context manager for database sessions"""
  session = Session(engine)
  try:
    yield session
    session.commit()
  except Exception:
    session.rollback()
    raise
  finally:
    session.close()


def get_db() -> Generator[Session, None, None]:
  """FastAPI dependency for database sessions"""
  with get_db_session() as session: yield session


class DatabaseMixin:
  """Mixin for database operations with consistent error handling"""

  @classmethod
  def get_by_id_or_404(cls, id: int, db: Session):
    """Get by ID or raise 404"""
    from app.exceptions import NotFoundException

    result = cls.get_by_id(id, db)
    if not result:
      raise NotFoundException(cls.__name__, str(id))
    return result

  @classmethod
  def get_by_name_or_404(cls, name: str, db: Session):
    """Get by name or raise 404"""
    from app.exceptions import NotFoundException

    result = cls.get_by_name(name, db)
    if not result:
      raise NotFoundException(cls.__name__, name)
    return result

  def save_with_rollback(self, db: Session):
    """Save with automatic rollback on error"""
    try:
      self.save(db)
    except Exception:
      db.rollback()
      raise
