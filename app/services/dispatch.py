"""Satu pintu masuk inference untuk semua task type.

Dipakai route `/models/auto-inference` dan stream realtime, yang tahu dataset-nya
tapi tidak perlu tahu task apa yang menanganinya. Ini satu-satunya modul service
yang memang lintas task — sisanya berdiri sendiri per task.
"""


def auto_inference(dataset, logger):
  """Jalankan inference sesuai task type dataset.

  TimeSeries dan Clustering diimpor di dalam fungsi, bukan di atas: keduanya
  menarik modul PyCaret-nya saat diimpor, jadi impor di level modul akan
  memperlambat boot service untuk task yang tidak memakainya.
  """
  if dataset.task_type.is_timeseries():
    from app.services import timeseries as timeseries_service
    return timeseries_service.auto_inference(dataset, logger)
  if dataset.task_type.is_clustering():
    from app.services import clustering as clustering_service
    return clustering_service.auto_inference(dataset, logger)
  if dataset.task_type.is_anomaly():
    from app.services import anomaly as anomaly_service
    return anomaly_service.auto_anomaly_detection(dataset, logger)

  from app.services import supervised as supervised_service
  return supervised_service.auto_predictions(dataset, logger)
