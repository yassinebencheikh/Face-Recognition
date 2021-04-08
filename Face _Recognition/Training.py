# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 12:32:24 2020

@author: Abdellah-Bencheikh
"""

from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
import cv2

# Chemin des dossiers
path_input_data = 'C:\\Users\\Abdellah-Bencheikh\\Desktop\\Mini_Projet_VA\\data'
path_output = 'C:\\Users\\Abdellah-Bencheikh\\Desktop\\Mini_Projet_VA\\Output'
Rotation_270=['BEALLAL Hafsa','BEN-FARES Anas','Bencheikh yassine','BINAN Lamiae','CHANAA Hiba','DLIA Asmae','EL MANSOURI Yousra','EL-HADDAR Besma','EL-HADDAR Nesma','ELBOUADI Mohammed','KANZOUAI Chaimae','MAHMOUDI Hanae','SAAOU MOHAMED','ZOUAK Firiel']
Sans_Rotation=['BENCHARFA Salma','BEL-LAHCEN Brahim','EL YASMI Zaineb']
Rotation_90=['MANESSOURI Meryem','KABIR Youssef','HAMDAOUI Abdessamad','ASSAMID Fatima-Ezzah','SADOUK Fadoua','KHIOUAH Asmae','OULED SIHAMMAN Noura']

def Training(path_input_data, path_output, n_neighbors=None):
    features = []
    labels = []
    folder=os.listdir(path_input_data)
    
    for label in folder:
        # Parcourez chaque image d'entraînement pour la personne 
        for video in os.listdir(os.path.join(path_input_data, label)):
            path_video=os.path.join(path_input_data,label, video)
            cap = cv2.VideoCapture(path_video)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if label in Rotation_270:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif label in Rotation_90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    
                # Ajouter l'encodage du visage pour l'image actuelle à l'ensemble d'entraînement    
                face_lacations = face_recognition.face_locations(frame)
                features.append(face_recognition.face_encodings(frame, known_face_locations=face_lacations)[0])
                labels.append(label)
                
    # Créer et former le classifieur KNN 
    Knn_Model = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    Knn_Model.fit(features, labels)
    
    # enregistrer data
    pickle_out = open(path_output +'\\data.pk', "wb")
    pickle.dump(features, pickle_out)
    pickle_out.close()
    # enregistrer Label
    pickle_out = open(path_output +'\\Labels.pk', "wb")
    pickle.dump(labels, pickle_out)
    pickle_out.close()
    # enregistrer Model
    pickle_out = open(path_output +'\\Model_test.pk', "wb")
    pickle.dump(Knn_Model, pickle_out)
    pickle_out.close()


def main(): 
    
    Training(path_input_data, path_output, n_neighbors=1)

if __name__=='__main__':
     main()



""" 
n_neighbors: (facultatif) nombre de voisins à peser dans la classification. Choisi automatiquement si non spécifié
knn_algo: (facultatif) la structure de données sous-jacente pour prendre en charge knn.default est ball_tree
verbose: verbosité de l'entraînement
"""

"""
Description de l'algorithme:
Le classificateur tricoté est d'abord formé sur un ensemble de visages étiquetés (connus) et peut ensuite prédire la personne
dans une image inconnue en trouvant les k visages les plus similaires (images avec des traits de placard sous la distance euclédienne)
dans son ensemble de formation, et effectuer un vote majoritaire (éventuellement pondéré) sur leur étiquette.


Par exemple, si k = 3, et les trois images de visage les plus proches de l'image donnée dans l'ensemble d'apprentissage sont une image de Biden
et deux images d'Obama, le résultat serait «Obama».

* Cette implémentation utilise un vote pondéré, de telle sorte que les votes des voisins les plus proches sont pondérés plus fortement.


Usage:
1. Préparez un ensemble d'images des personnes connues que vous souhaitez reconnaître. Organisez les images dans un seul répertoire
   avec un sous-répertoire pour chaque personne connue.
2. Ensuite, appelez la fonction «train» avec les paramètres appropriés. Assurez-vous de passer le 'model_save_path' si vous
   souhaitez enregistrer le modèle sur le disque afin de pouvoir réutiliser le modèle sans avoir à le réentraîner.
"""