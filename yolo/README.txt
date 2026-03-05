== YOLOv8 — Detection de pistes ==

1. Placer le modele (.pt) dans yolo/
2. Placer les images (.jpeg) dans yolo/test_images/test/
3. Lancer la prediction :

   python yolo/predict.py -n 100        # 100 premieres images
   python yolo/predict.py               # toutes les images
   python yolo/predict.py --conf 0.5    # seuil de confiance a 0.5

4. Resultats dans yolo/output/ :
   - labels/   → fichiers .txt (classe, bbox, confiance par image)
   - images annotees avec les bboxes dessinees

Note : le modele (.pt), les images et les outputs sont dans le .gitignore.
