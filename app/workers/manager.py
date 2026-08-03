import subprocess,json,sys
from typing import List,Dict,Optional,Union
from app.logger  import LOGGER_GLOBAL
from app.database.schemas import TaskType,StatusProcess
from app.config import Config
from app.database.base import get_session
from app.database.DB import Dataset
from contextlib import closing
from app import exceptions


def _run(text:List[str]) -> str:
  try:
    command = subprocess.run(text,capture_output=True,text=True,check=True)
    return command.stdout
  except subprocess.CalledProcessError as e :
    raise exceptions.InvalidStateException("Task",str(e),"Stopping")
  except FileNotFoundError :
    outs = f"command not found " + "".join(text)
    LOGGER_GLOBAL.error(outs)
    raise exceptions.NotFoundException("Task",outs)

class WorkerManager:
  @staticmethod
  def get_script_path(task_type:TaskType):
    # Setiap task punya file entry sendiri di service layer (core-nya murni,
    # tidak punya main). File inilah identitas worker: sebuah worker Anomaly
    # yang menumpang dataset Regression dikenali dari skrip yang dijalankan,
    # bukan dari argumen tambahan yang bisa tertukar.
    if task_type.is_supervised(): script_path = Config.dir_app/"services"/"supervised.py"
    elif task_type.is_unsupervised(): script_path = Config.dir_app/"services"/"clustering.py"
    elif task_type.is_timeseries(): script_path = Config.dir_app/"services"/"timeseries"/"worker.py"
    else: script_path = Config.dir_app/"services"/"anomaly.py"
    return script_path

  @staticmethod
  def update_flag_dataset(dataset_name:str,is_stop:bool=False):
    with closing(next(get_session())) as db:
      dataset = Dataset.get_by_name(dataset_name,db)
      if not dataset: return 
      if is_stop: dataset.status = StatusProcess.IDLE
      else: dataset.status = StatusProcess.ACTIVE
      dataset.save(db)

  @staticmethod
  def start(task_name:str):
    task = WorkerManager.get_tasks(task_name)
    command = ["pm2","start",task_name]
    _run(command)
    dataset_name = task.get("dataset_name","")
    WorkerManager.update_flag_dataset(dataset_name,is_stop=False)
    LOGGER_GLOBAL.info(f"Task {task_name} is start ...")
    

  @staticmethod
  def create(dataset_name:str,task_type:TaskType):
    task_name = f"{dataset_name}-{str(task_type)}"
    script_path = WorkerManager.get_script_path(task_type)
    # Argv cukup nama dataset: task type sudah tersirat dari skrip yang dipilih
    # get_script_path.
    command = ["pm2","start",str(script_path),
              "--name",task_name,"--interpreter",sys.executable,
              "--interpreter-args", "-u","--",dataset_name]

    _run(command)
    if task_type != TaskType.Anomaly: WorkerManager.update_flag_dataset(dataset_name,is_stop=False)
    LOGGER_GLOBAL.info(f"script : {script_path}")
    LOGGER_GLOBAL.info(f"Task {task_name} is start ...")

  @staticmethod
  def stop(task_name:str):
    task = WorkerManager.get_tasks(task_name)
    dataset_name = task.get("dataset_name","")
    command = ["pm2","stop",task_name]
    _run(command)
    if "Anomaly" not in task_name: WorkerManager.update_flag_dataset(dataset_name,is_stop=True)
    LOGGER_GLOBAL.info(f"task {task_name} is stopping ...")

  @staticmethod
  def delete(task_name:str):
    task = WorkerManager.get_tasks(task_name)
    dataset_name = task.get("dataset_name","")
    command = ["pm2","delete",task_name]
    _run(command)
    WorkerManager.delete_logs(task_name)
    WorkerManager.update_flag_dataset(dataset_name,is_stop=True)
    LOGGER_GLOBAL.info(f"Task {task_name} is deleting")

  @staticmethod
  def delete_by_dataset(dataset_name:str):
    tasks = WorkerManager.get_tasks()
    for d in tasks:
      if d['dataset_name'] == dataset_name:
        task_name = d["name"]
        WorkerManager.delete(task_name)
    LOGGER_GLOBAL.info(f"Successfully deleted all task datasets {dataset_name}")

  @staticmethod
  def delete_logs(task_name:str):
    log_path = f"~/.pm2/logs/{task_name}*"
    _run(["pm2","flush",task_name])
    _run(["rm","-f",log_path])

  @staticmethod
  def get_log_task(task_name:str,is_err:bool=False):
    if is_err: log_path = f"/home/imaji/.pm2/logs/{task_name}-error.log"
    else: log_path = f"/home/imaji/.pm2/logs/{task_name}-out.log"
    try:
      with open(log_path,'r') as f: outs = f.read()
      return outs
    except FileNotFoundError: pass

  @staticmethod
  def get_log_worker():
    log_path = "/home/imaji/.pm2/pm2.log"
    try:
      with open(log_path,'r') as f: outs = f.read()
      return outs
    except FileNotFoundError: pass

  @staticmethod
  def get_tasks(task_name:Optional[str]=None) -> Union[List[Dict],Dict]:
    result = subprocess.run(
        ['pm2', 'jlist'],
        capture_output=True,
        text=True,
        check=True
    )
    # Parse JSON
    processes = json.loads(result.stdout)
    cleaned = []
    for proc in processes:
      dataset = "-".join(proc.get("name").split("-")[:2])
      task_type = proc.get("name").split("-")[-1]
      task = {
          'name': proc.get('name'),
          "dataset_name":dataset,
          'task_type':task_type,
          'pm_id': proc.get('pm_id'),
          'pid': proc.get('pid'),
          'status': proc.get('pm2_env', {}).get('status'),
          'restarts': proc.get('pm2_env', {}).get('restart_time'),
          'uptime': proc.get('pm_uptime'),
          'memory': proc.get('monit', {}).get('memory'),
          'cpu': proc.get('monit', {}).get('cpu'),
          'mode': proc.get('pm2_env', {}).get('exec_mode'),
          'created_at': proc.get('pm2_env', {}).get('created_at'),
      }
      if task_name and task["name"] == task_name: return task
      if task['name'] in ["agent-service","frontend-v1","main-service"]: continue
      else: cleaned.append(task)
    if task_name: return []
    else: return cleaned
  
  @staticmethod
  def is_active(task_name:str) -> bool:
    task = WorkerManager.get_tasks(task_name)
    if not task: return False
    return True if task.get("status",False) == "online" else False
    
if __name__ == '__main__':
  import pprint
  dataset_name = "Regression-06eafb4a"
  task_type = TaskType.Regression
  # WorkerState.stop(task_name)
  # pprint.pprint(WorkerManager.get_tasks())
  a = WorkerManager.create(dataset_name,task_type)
  print(a)
  
  


