import os
import shutil
from pathlib import Path
from contextlib import closing
from app.database.db import get_session, engine, DB_PATH
from app.database.orm import Dataset, ModelML
from app.config import Config
from sqlmodel import delete, select


def clean_all_data():
    """
    Menghapus semua data dari database beserta file model dan storage.
    """
    print("=" * 60)
    print("MEMBERSIHKAN SEMUA DATA DAN FILE MODEL")
    print("=" * 60)

    with closing(next(get_session())) as db:
        # 1. Ambil semua dataset untuk mendapatkan path folder yang akan dihapus
        datasets = db.exec(select(Dataset)).all()

        if datasets:
            print(f"\nMenemukan {len(datasets)} dataset untuk dihapus...")

            # 2. Hapus folder storage untuk setiap dataset
            for dataset in datasets:
                storage_path = Config.dir / "storages" / dataset.name
                if os.path.exists(storage_path):
                    print(f"  - Menghapus folder storage: {storage_path}")
                    shutil.rmtree(storage_path)
                else:
                    print(f"  - Folder storage tidak ditemukan: {storage_path}")
        else:
            print("\nTidak ada dataset yang ditemukan di database.")

        # 3. Hapus semua data dari tabel ModelML
        print("\nMenghapus semua record dari tabel ModelML...")
        result = db.exec(delete(ModelML))
        db.commit()
        print(f"  - {result.rowcount} record ModelML dihapus")

        # 4. Hapus semua data dari tabel Dataset
        print("\nMenghapus semua record dari tabel Dataset...")
        result = db.exec(delete(Dataset))
        db.commit()
        print(f"  - {result.rowcount} record Dataset dihapus")

    print("\n" + "=" * 60)
    print("SEMUA DATA BERHASIL DIHAPUS!")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print(f"Storage folder: {Config.dir / 'storages'}")
    print("=" * 60)


def clean_database_only():
    """
    Hanya menghapus data dari database tanpa menghapus file storage.
    """
    print("=" * 60)
    print("MEMBERSIHKAN HANYA DATA DATABASE")
    print("=" * 60)

    with closing(next(get_session())) as db:
        # Hapus semua data dari tabel ModelML
        print("\nMenghapus semua record dari tabel ModelML...")
        result = db.exec(delete(ModelML))
        db.commit()
        print(f"  - {result.rowcount} record ModelML dihapus")

        # Hapus semua data dari tabel Dataset
        print("\nMenghapus semua record dari tabel Dataset...")
        result = db.exec(delete(Dataset))
        db.commit()
        print(f"  - {result.rowcount} record Dataset dihapus")

    print("\n" + "=" * 60)
    print("DATA DATABASE BERHASIL DIHAPUS!")
    print("=" * 60)


def clean_storage_only():
    """
    Hanya menghapus folder storage tanpa menghapus data database.
    """
    print("=" * 60)
    print("MEMBERSIHKAN HANYA FOLDER STORAGE")
    print("=" * 60)

    with closing(next(get_session())) as db:
        datasets = db.exec(select(Dataset)).all()

        if datasets:
            print(f"\nMenemukan {len(datasets)} dataset...")

            for dataset in datasets:
                storage_path = Config.dir / "storages" / dataset.name
                if os.path.exists(storage_path):
                    print(f"  - Menghapus folder: {storage_path}")
                    shutil.rmtree(storage_path)
                else:
                    print(f"  - Folder tidak ditemukan: {storage_path}")
        else:
            print("\nTidak ada dataset yang ditemukan.")

    print("\n" + "=" * 60)
    print("FOLDER STORAGE BERHASIL DIHAPUS!")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "--db-only":
            clean_database_only()
        elif command == "--storage-only":
            clean_storage_only()
        elif command == "--all" or command == "-a":
            clean_all_data()
        elif command == "--help" or command == "-h":
            print("""
Penggunaan: python clean_database.py [opsi]

Opsi:
  --all, -a          Hapus semua data database dan folder storage (default)
  --db-only          Hanya hapus data dari database
  --storage-only     Hanya hapus folder storage
  --help, -h         Tampilkan bantuan ini

Contoh:
  python clean_database.py              # Hapus semua data
  python clean_database.py --db-only    # Hanya hapus database
  python clean_database.py --storage-only  # Hanya hapus storage
            """)
        else:
            print(f"Opsi tidak dikenal: {command}")
            print("Gunakan --help untuk melihat bantuan")
    else:
        # Default: hapus semua
        clean_all_data()
