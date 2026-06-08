import logging
from pathlib import Path
from sqlmodel import SQLModel, create_engine, Session
from contextlib import contextmanager

from app.database.orm import Dataset, ModelML

# Gunakan path absolut yang lebih aman
BASE_DIR = Path(__file__).parent.parent.parent
DB_DIR = BASE_DIR / "storages" / "rtdb"
DB_PATH = DB_DIR / "data.db"

# Pastikan direktori database ada dan writable
DB_DIR.mkdir(parents=True, exist_ok=True)

# Gunakan absolute path untuk SQLite
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Buat engine global dengan check_same_thread=False untuk FastAPI
engine = create_engine(
    DATABASE_URL, echo=False, connect_args={"check_same_thread": False}
)

# Konfigurasi logging untuk SQLAlchemy
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_session():
  """Dependency untuk mendapatkan session database."""
  with Session(engine) as session:
    yield session

@contextmanager
def get_db_session():
  session_gen = get_session()
  db = next(session_gen)
  try: yield db
  finally:
    try: next(session_gen)  # Tutup generator dengan benar
    except StopIteration: pass


def init_db(drop_existing: bool = False):
  """Inisialisasi database: membuat tabel sesuai model SQLModel."""
  print(f"Inisialisasi database di: {DB_PATH}")

  if drop_existing:
    print("Menghapus tabel lama...")
    SQLModel.metadata.drop_all(engine)

  print("Membuat tabel baru (jika belum ada)...")
  SQLModel.metadata.create_all(engine)

  print("Tabel selesai dibuat.")


if __name__ == "__main__":
  init_db()
