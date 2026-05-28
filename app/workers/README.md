# Worker System

Sistem worker untuk menjalankan ML inference secara background dengan fitur auto-reload model, backup & rollback, dan monitoring.

# Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        WorkerManager                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Worker 1    │  │ Worker 2    │  │ Worker N    │             │
│  │(asyncio)   │  │(asyncio)   │  │(asyncio)   │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
└─────────┼────────────────┼────────────────┼────────────────────┘
          │                │                │
          └────────────────┴────────────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
     ┌──────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
     │  InfluxDB   │ │ SQLite   │ │ Local Files │
     │  (Results)  │ │ (State)  │ │ (Logs/Queue)│
     └─────────────┘ └──────────┘ └─────────────┘
```

## Components

### 1. WorkerManager (`manager.py`)
Orchestrator utama untuk semua workers.

**Features:**
- Multiple workers concurrent (asyncio)
- Thread pool untuk ML inference (CPU-bound)
- Batch writing ke InfluxDB
- Auto-recovery saat restart
- Graceful shutdown

**Key Methods:**
```python
manager = get_worker_manager()
await manager.start_worker("dataset_name")
await manager.stop_worker("dataset_name", graceful=True)
await manager.restart_worker("dataset_name")
```

### 2. WorkerStateManager (`state_manager.py`)
SQLite persistence sebagai single source of truth.

**State Table:**
- `dataset_name` (PK)
- `status`: idle, running, paused, error, stopped, training, reload_required
- `model_version` & `latest_model_version`
- `total_predictions`: Counter prediksi berhasil (hanya yang ter-write ke InfluxDB)
- `error_count`, `consecutive_errors`
- `last_run_at`, `started_at`

**Key Methods:**
```python
state_manager.create_worker(dataset_name, task_type, interval_seconds)
state_manager.update_status(dataset_name, WorkerStatus.RUNNING)
state_manager.update_metrics(dataset_name, predictions_count=1)
state_manager.delete_worker(dataset_name)  # Hard delete
```

### 3. ModelBackupManager (`model_backup.py`)
Backup dan versioning model.

**Features:**
- Auto-backup sebelum training (max 3 backup)
- Atomic model swap (rename file)
- Rollback ke versi sebelumnya
- Version tracking

**Key Methods:**
```python
backup_manager.backup_model(dataset_name, model_name, model_path, db)
backup_manager.restore_model(dataset_name, model_name, version, target_path, db)
version_manager.increment_version(dataset_name, db)
```

### 4. Training Integration (`training_integration.py`)
Notifikasi antara training dan worker.

**Flow:**
```
Training Start  → Worker pakai model lama
Training Complete → Backup old model → Increment version 
                  → Notify worker → Worker reload model baru
```

**Usage:**
```python
training_notifier.notify_training_start("dataset_name")
training_notifier.notify_training_complete("dataset_name", "model_name", model_path)
```

### 5. PredictionWorkerLogger (`prediction_logger.py`)
Logger terpisah per dataset.

**Log Location:**
```
storages/<dataset_name>/logs/worker.log
```

**Features:**
- Log ke file terpisah per dataset
- Format readable dengan emoji
- DEBUG ke file, INFO ke console

**Usage:**
```python
pw_logger = get_prediction_worker_logger("prediction_worker:dataset_name", "dataset_name")
pw_logger.log_worker_start("dataset_name")
pw_logger.log_prediction_summary(iteration, success_count, error_count, total_count)
```

### 6. Local Queue (`influx_queue.py`)
SQLite queue untuk failed InfluxDB writes.

**Features:**
- Persistent storage (survives restart)
- Automatic retry dengan exponential backoff
- Priority queue (older items first)
- Max 5 retry attempts

**Location:** `queue.db` di folder storages/

## Workflow

### Prediction Flow
```
1. Worker loop start (setiap interval)
2. Load dataset & models dari DB
3. Run inference untuk setiap active model
4. Fetch actual value (dari hasil inference)
5. Run anomaly detection (jika enabled)
6. Write ke InfluxDB (batch)
   - Sukses: total_predictions += success_count
   - Gagal: Masuk local queue untuk retry
7. Update metrics di SQLite
8. Sleep sampai next interval
```

### Training Flow
```
1. notify_training_start()
   - Worker tetap pakai model lama
   
2. Training berjalan...

3. notify_training_complete()
   - Backup old model
   - Increment model version
   - Set worker status = RELOAD_REQUIRED
   
4. Worker iteration berikutnya:
   - Detect status RELOAD_REQUIRED
   - Reload model baru
   - Update status = RUNNING
```

## API Endpoints

### Lifecycle
```http
POST   /workers/{dataset}/start
POST   /workers/{dataset}/stop?graceful=true
POST   /workers/{dataset}/restart
POST   /workers/{dataset}/pause
POST   /workers/{dataset}/resume
DELETE /workers/{dataset}?delete_influx_data=true&graceful_timeout=30
```

### Status & Monitoring
```http
GET /workers/status
GET /workers/{dataset}/status
GET /workers/{dataset}/model-version
GET /workers/{dataset}/logs?limit=100
```

### Predictions
```http
GET /workers/{dataset}/predictions?hours=24&limit=1000
GET /workers/{dataset}/predictions/latest
GET /workers/{dataset}/predictions/stats?hours=24
```

### Anomaly Detection
```http
POST /workers/{dataset}/anomaly/enable
POST /workers/{dataset}/anomaly/disable
GET  /workers/{dataset}/anomaly/status
```

### Model Backup
```http
GET  /workers/{dataset}/backups
POST /workers/{dataset}/rollback?version=1
```

### Batch Operations
```http
POST /workers/batch/start
POST /workers/batch/stop
```

### Admin
```http
GET  /workers/admin/stats
POST /workers/admin/cleanup?max_age_hours=24
```

## Database Schema

### WorkerState (worker_states)
```sql
CREATE TABLE worker_states (
    dataset_name VARCHAR PRIMARY KEY,
    status VARCHAR NOT NULL,  -- enum: idle, running, paused, error, stopped, training, reload_required
    task_type VARCHAR NOT NULL,
    model_version INTEGER DEFAULT 1,
    latest_model_version INTEGER DEFAULT 1,
    interval_seconds INTEGER DEFAULT 300,
    total_predictions INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    consecutive_errors INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    started_at TIMESTAMP,
    last_run_at TIMESTAMP,
    last_model_update TIMESTAMP,
    training_started_at TIMESTAMP,
    training_completed_at TIMESTAMP,
    extra_data JSON  -- {anomaly_enabled: bool, ...}
);
```

### ModelBackup (model_backups)
```sql
CREATE TABLE model_backups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    version INTEGER NOT NULL,
    backup_path VARCHAR NOT NULL,
    metadata JSON,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### WorkerEvent (worker_events)
```sql
CREATE TABLE worker_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,  -- worker_created, status_changed, training_started, etc
    event_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Configuration

**Environment Variables:**
```bash
# InfluxDB
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_token
INFLUXDB_BUCKET=ml-buckets
INFLUXDB_ORG=tech

# Worker
WORKER_MAX_WORKERS=4          # Thread pool size
WORKER_MAX_CONSECUTIVE_ERRORS=5  # Auto-stop threshold
WORKER_ERROR_BACKOFF_BASE=5   # Exponential backoff base (seconds)
```

**Config Class:**
```python
from app.config import Config

Config.dir                    # Base directory
Config.influxdb_token        # InfluxDB token
Config.interval              # Default prediction interval (minutes)
```

## Error Handling

### Consecutive Errors
- Worker akan auto-stop jika `consecutive_errors >= max_consecutive_errors` (default: 5)
- Exponential backoff antara retries: `min(5 * 2^n, 300)` detik

### InfluxDB Failures
- Failed writes masuk ke local queue (SQLite)
- Automatic retry dengan exponential backoff
- Max 5 retry attempts per item
- Items lama (>24 jam) dan >5 retries di-cleanup

### Graceful Degradation
- Jika InfluxDB down: data masuk queue, worker tetap jalan
- Jika model corrupt: gunakan backup/rollback
- Jika worker error: restart otomatis (jika configured)

## Monitoring

### Check Worker Health
```bash
# Status worker
curl http://localhost:8000/workers/my_dataset/status

# Log worker
tail -f storages/my_dataset/logs/worker.log

# Queue stats
sqlite3 storages/queue.db "SELECT * FROM pending_predictions;"
```

### Key Metrics
- `total_predictions`: Total berhasil write ke InfluxDB
- `error_count`: Total errors (termasuk yang di-retry)
- `consecutive_errors`: Errors berturut-turut (auto-stop threshold)
- Queue size: Items di local queue menunggu retry

## Best Practices

1. **Always graceful stop**: `graceful=true` untuk menghindari data loss
2. **Monitor consecutive_errors**: Setup alerting jika >3
3. **Regular cleanup**: Run `/workers/admin/cleanup` setiap hari
4. **Backup before training**: Auto-backup aktif, tapi verify backup exists
5. **Check model versions**: Pastikan `current_version == latest_version`

## Troubleshooting

### Worker running tapi tidak ada data di InfluxDB?
1. Check `consecutive_errors` di status
2. Check `queue.db` untuk failed writes
3. Check log: `tail -f storages/{dataset}/logs/worker.log`
4. Verify InfluxDB connection: `GET /workers/{dataset}/predictions/latest`

### Model tidak reload setelah training?
1. Check `GET /workers/{dataset}/model-version`
2. Verify `needs_reload: true`
3. Check next iteration logs untuk reload confirmation

### High error rate?
1. Check `GET /workers/{dataset}/logs` untuk error details
2. Verify dataset models masih valid
3. Check InfluxDB connectivity
4. Consider restart worker: `POST /workers/{dataset}/restart`

## File Structure

```
app/workers/
├── __init__.py              # Exports
├── manager.py               # WorkerManager
├── state_manager.py         # WorkerStateManager
├── model_backup.py          # Backup & versioning
├── training_integration.py  # Training hooks
├── prediction_logger.py     # Logger per dataset
├── lifecycle.py             # Startup/shutdown handlers
└── README.md                # This file
```

## Related Files

```
app/
├── database/
│   ├── worker_state.py      # SQLModel definitions
│   └── influx.py            # InfluxDB client
├── routes/
│   └── worker_routes.py     # API endpoints
└── utils/
    └── influx_queue.py      # Local queue untuk retries
```
