== YOLOv8 — Detection de pistes ==

Modele: yolo/yolov8nTest.pt (entraine sur LARD pour detection de pistes,
embarque avion). Inference offline via ultralytics.
Note: renomme depuis yolov8n.pt pour eviter qu'ultralytics auto-telecharge
le modele COCO de base (80 classes) si le fichier est introuvable.

Pipeline standard (recommande): voir run_pipeline.py a la racine.
La phase 3 (Detection_Evaluation.py) appelle predict_run + evaluate_run
sur chaque runs/<ICAO_RWY>/.

== Modules ==

  predict.py   predict_run(run_dir, conf, imgsz)
               -> runs/<run>/predictions.csv + predictions_txt/
               (lit degraded/ si fautes appliquees, sinon footage/)

  evaluate.py  evaluate_run(run_dir, runway, iou_thresh, iou_method)
               -> dict de metriques (AP, F1, P, R, TP, FP, FN)
               (lit predictions.csv + *_labels.csv du run)

  eval/        Module bas-niveau (box.py, metrics.py, metrics_utils.py)
               -- conversions bbox + IoU torch (IOU/GIOU/DIOU/CIOU).

  camera_sensor_errors/
               22 fonctions de degradation OpenCV (apply_errors).
               Appele depuis project/export/sensor_faults.py.

== Usage CLI standalone ==

  py yolo/predict.py --images <dir> --output <dir> --conf 0.25
  py yolo/evaluate.py --predictions runs/<run>/predictions.csv \
                      --csv runs/<run>/<run>_labels.csv --runway 24

== Metriques (compute_metrics) ==

  AP        Average Precision (aire sous courbe precision-recall)
  F1        Score F1 (moyenne harmonique precision/recall)
  P         TP / (TP + FP)
  R         TP / (TP + FN)
  TP/FP/FN  comptes au seuil IoU donne (defaut 0.5, methode CIOU)

Note: les CSV LARD listent toutes les pistes visibles a l'aeroport.
evaluate.py filtre via --runway (extrait du nom du run par defaut),
sinon les pistes non detectees comptent comme FN et ecrasent le recall.
