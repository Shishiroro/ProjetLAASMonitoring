== YOLOv8 — Detection de pistes ==

1. Placer le modele (.pt) dans yolo/
2. Placer les images (.jpeg) dans yolo/test_images/test/
3. Lancer la prediction :

   python yolo/predict.py -n 100        # 100 premieres images
   python yolo/predict.py               # toutes les images
   python yolo/predict.py --conf 0.5    # seuil de confiance a 0.5

4. Resultats dans yolo/output/expN/ :
   - labels/   → fichiers .txt (classe, bbox, confiance par image)
   - images annotees avec les bboxes dessinees

5. Prediction sur un dataset specifique :

   python yolo/predict.py --images yolo/datasetsGES/LFPO_24/footage

   Sans --images, ca pointe toujours sur test_images/test/ par defaut.

== Pipeline complet (generation → evaluation) ==

1. Generer un scenario : python main.py (ou python project/Generate.py)
   → produit .esp + .yaml dans output/

2. Importer le .esp dans Google Earth Studio → rendre les images

3. Preparer le dataset :
   - Creer yolo/datasetsGES/<AIRPORT_RUNWAY>/
     Convention de nommage : <ICAO>_<RWY> (ex: LFPO_24, LFPG_09R)
     IMPORTANT : le nom du dossier doit correspondre au prefixe des images
     (LARD cherche <dossier_stem>_000.jpeg, <dossier_stem>_001.jpeg, ...)
   - Placer les images GES dans yolo/datasetsGES/<AIRPORT_RUNWAY>/footage/
   - Copier le .yaml depuis output/ vers yolo/datasetsGES/<AIRPORT_RUNWAY>/

4. Generer le CSV ground truth (apres avoir les images) :

   python yolo/generate_gt.py --yaml yolo/datasetsGES/LFPO_24/LFPO_24.yaml --dataset yolo/datasetsGES/LFPO_24

   Note : generate_gt.py desactive le filtre d'orientation LARD pour
   labelliser TOUTES les pistes visibles (y compris la piste d'approche).
   Le CSV contient une ligne par piste visible par image (colonne "runway").

5. Lancer YOLO :

   python yolo/predict.py --images yolo/datasetsGES/LFPO_24/footage --conf 0.1

6. Evaluer :

   # Evaluer sur la piste d'approche uniquement (recommande) :
   python yolo/evaluate.py --labels yolo/output/expN/labels --csv yolo/datasetsGES/LFPO_24/LFPO_24_labels.csv --runway 24

   # Evaluer sur toutes les pistes visibles :
   python yolo/evaluate.py --labels yolo/output/expN/labels --csv yolo/datasetsGES/LFPO_24/LFPO_24_labels.csv

   Le filtre --runway est important : le CSV contient toutes les pistes de
   l'aeroport (ex: 6 pistes a LFPO), mais YOLO ne detecte en general qu'une
   seule piste par image. Sans filtre, les pistes non detectees comptent
   comme FN et ecrasent le recall.

   Resultats sauvegardes dans yolo/output/expN/eval_results.json

== Options evaluate.py ==

   --labels    Dossier des .txt predictions YOLO (requis)
   --csv       Fichier .csv ground truth LARD (requis)
   --runway    Filtrer GT sur une piste (ex: 24, 09R). Defaut: toutes.
   --iou-thresh  Seuil IoU pour matching (defaut: 0.5)
   --iou-method  Methode IoU : IOU, GIOU, DIOU, CIOU (defaut: CIOU)

== Metriques ==

   AP        Average Precision (aire sous courbe precision-recall)
   F1        Score F1 (moyenne harmonique precision/recall)
   Precision TP / (TP + FP) — proportion de detections correctes
   Recall    TP / (TP + FN) — proportion de GT detectees
   TP        True Positives (prediction matche un GT avec IoU >= seuil)
   FP        False Positives (prediction sans GT correspondant)
   FN        False Negatives (GT sans prediction correspondante)

== Notes ==

- Convention coordonnees LARD : x=vertical, y=horizontal (inverse du standard).
  evaluate.py gere le swap automatiquement.
- Le modele (.pt), les images et les outputs sont dans le .gitignore.
- imgsz=512 par defaut dans predict.py (coherent avec l'entrainement YOLO).
  Les images sont redimensionnees en interne pour l'inference, les labels
  sont en coordonnees normalisees (0-1), independantes de la resolution source.
