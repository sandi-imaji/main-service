"""app.workers.manager.WorkerManager.create — penyusunan perintah pm2.

Identitas worker ditentukan oleh SKRIP yang dijalankan, bukan argumen tambahan.
Ini yang mengunci perbaikan worker anomaly di sisi spawn: sebuah worker Anomaly
yang menumpang dataset Regression harus menjalankan `services/anomaly.py`, bukan
entry task utama dataset-nya.
"""
from app.database.schemas import TaskType
from app.workers.manager import WorkerManager
from app.workers import manager as manager_module


def _capture_run(monkeypatch):
  """Replace _run so create() builds-but-does-not-execute the pm2 command."""
  captured = {}
  monkeypatch.setattr(manager_module, "_run", lambda cmd: captured.setdefault("cmd", cmd))
  monkeypatch.setattr(WorkerManager, "update_flag_dataset",
                      staticmethod(lambda *a, **k: None))
  return captured


class TestScriptPath:
  """Tiap task type menunjuk entry-nya sendiri, dan file itu harus benar-benar ada
  — salah path baru ketahuan saat worker gagal start di produksi."""

  def test_each_task_type_has_an_existing_entry(self):
    for task_type in TaskType:
      path = WorkerManager.get_script_path(task_type)
      assert path.exists(), f"entry worker {task_type.name} tidak ada: {path}"

  def test_anomaly_entry_differs_from_supervised(self):
    """Keduanya dulu file yang sama (services/inference.py) sehingga identitas
    worker harus dioper lewat argv."""
    assert WorkerManager.get_script_path(TaskType.Anomaly) \
        != WorkerManager.get_script_path(TaskType.Regression)

  def test_supervised_share_one_entry(self):
    assert WorkerManager.get_script_path(TaskType.Regression) \
        == WorkerManager.get_script_path(TaskType.Classification)


class TestCreateCommand:
  def test_anomaly_worker_runs_the_anomaly_entry(self, monkeypatch):
    captured = _capture_run(monkeypatch)
    WorkerManager.create("Regression-test", TaskType.Anomaly)
    cmd = captured["cmd"]
    # dataset-nya Regression, tapi skrip yang dijalankan harus anomaly
    assert str(WorkerManager.get_script_path(TaskType.Anomaly)) in cmd
    assert cmd[-1] == "Regression-test"          # argv terakhir: nama dataset

  def test_dataset_name_is_the_only_argv(self, monkeypatch):
    captured = _capture_run(monkeypatch)
    WorkerManager.create("Regression-test", TaskType.Regression)
    cmd = captured["cmd"]
    # setelah "--" hanya ada satu argumen: nama dataset
    assert cmd[cmd.index("--") + 1:] == ["Regression-test"]

  def test_create_names_task_by_dataset_and_type(self, monkeypatch):
    captured = _capture_run(monkeypatch)
    WorkerManager.create("Regression-test", TaskType.Anomaly)
    cmd = captured["cmd"]
    # pm2 --name <dataset>-<task_type>
    assert "Regression-test-Anomaly" in cmd
