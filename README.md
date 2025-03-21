# INF5109 : Vision Par Ordinateur - Groupe 10
## Theme:Détection d'objets et de segmentation d'images basée sur Mask R-CNN
## Article Selectionné
# Transfer Learning pour la Segmentation d'Instances de Bouteilles avec Mask R-CNN

Ce notebook reproduit le code d’un article pour détecter et segmenter des bouteilles plastiques via transfer learning avec Mask R-CNN, dans le but d’automatiser le recyclage. Réalisé sur Kaggle, il est divisé en 2 parties.  
[Lien du notebook Kaggle](https://www.kaggle.com/code/franckvalere/inf5109-vision-par-ordinateur-groupe-10/edit)

**Auteurs** : Tamwo Feuwo Franck Valère - 20u2837, Kuetche Ngoufack Fideline - 20u2958

---

## Préambule : Problème de Compatibilité avec TensorFlow et Solution

Mask R-CNN requiert une version de TensorFlow inférieure ou égale à 1.15.0. Cependant, les environnements d’exécution actuels ne prennent plus en charge cette version, et celle-ci a été retirée des dépôts officiels, rendant la reproduction de l’approche initiale très difficile. Pour contourner ce problème, nous avons utilisé un environnement d’exécution récupéré via Kaggle, basé sur le code de "Ashraf Khan" ([lien](https://www.kaggle.com/code/ashrafkhan94/matterport-mask-r-cnn-model-object-detection)), datant du 11 février 2021. Cet environnement inclut les dépendances nécessaires, notamment TensorFlow 1.15.0, et a permis de reproduire fidèlement le code de l’article sélectionné.

---

## Exécution Interactive sur Kaggle
- Ouvrez le fichier notebook pour visualiser le travail effectuer
- Pour exécuter ce notebook de manière interactive avec toutes les dépendances préconfigurées, nous vous invitons à vous rendre sur Kaggle. Le notebook est disponible dans le dépôt suivant : [INF5109 - Vision Par Ordinateur - Groupe 10](https://www.kaggle.com/code/franckvalere/inf5109-vision-par-ordinateur-groupe-10/edit). Si vous avez été ajouté comme collaborateur à ce dépôt, vous pouvez directement lancer et modifier le notebook en ligne. Cela garantit une exécution fluide sans avoir à recréer localement l’environnement spécifique requis. et un suivi fluide des instructions mentionnees ici

---

## Partie 1 : Transfer Learning avec COCO et Réentraînement

- **Objectif** : Utiliser Mask R-CNN pour détecter des bouteilles plastiques.
- **Dataset** : 127 images collectées.
- **Étapes** :
  - Configuration de l’environnement :
    ```bash
    cd /kaggle/input/mask-rcnn
    !pip3 install -r requirements.txt
    cp -r /kaggle/input/mask-rcnn/Mask_RCNN /kaggle/working/
    cd /kaggle/working/Mask_RCNN/Mask_RCNN
    !python setup.py install
    ```
  - Chargement des poids pré-entraînés de COCO (inclut la classe "bottle").
  - Exécuter les cellules 8 à 16 pour prédire sur le dataset.
  - Résultat : Détection imprécise des bouteilles (voir image des résultats).
- **Réentraînement** :
  - Réentraînement des couches "heads" de Mask R-CNN avec cette configuration :
    ```python
    class BottleConfig(Config):
        NAME = "bottle"
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 2  # Background + bottle
        STEPS_PER_EPOCH = 1000
        DETECTION_MIN_CONFIDENCE = 0.9
    ```
  - Utilisation des poids de COCO comme base.
  - Sauvegarde des poids entraînés à : `/kaggle/input/bottle_weight_model/tensorflow1/default/1/mask_rcnn_bottle_0100.h5`.

---

## Partie 2 : Inférence

- **Objectif** : Charger les poids entraînés et prédire sur images/vidéos.
- **Étapes** :
  - Chargement des poids sauvegardés : `/kaggle/input/bottle_weight_model/tensorflow1/default/1/mask_rcnn_bottle_0100.h5`.
  - Passage le modèle en mode inférence.
  - Exécuter les cellules de la section "Inférence".
- **Prédiction sur images** :
  - Choisir une image aléatoire dans :
    - `/kaggle/input/bottle-images/images` (ex. : `/kaggle/input/bottle-images/images/test4.jpg`)
    - Ou `/kaggle/input/bottle-images/dataset/val`.
  - Mettre à jour le chemin dans `detect_and_display_images()`.
- **Prédiction sur vidéos** :
  - Utiliser une vidéo dans `/kaggle/input/demo-video` (ex. : `/kaggle/input/demo-video/part2(split-video.com).mp4`).
  - Mettre à jour le chemin dans la fonction correspondante.

---

## Notes
- Prioriser l’exécution cellule par cellule.
- Les poids sauvegardés évitent de réentraîner le modèle à chaque fois.
- Tout est prêt, pas de configuration requise.
- Modifiez uniquement les chemins des images/vidéos pour tester d’autres exemples.
- Résultats visibles en exécutant cellule par cellule.
