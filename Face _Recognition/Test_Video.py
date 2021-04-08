# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 13:42:19 2020

@author: Abdellah-Bencheikh
"""

import face_recognition
import cv2
import pickle

# Chargement du Modele
path_model = 'C:\\Users\\Abdellah-Bencheikh\\Desktop\\Mini_Projet_VA\\Output\\Model.pk'
file = open(path_model,'rb')
Model = pickle.load(file)

def Prediction(frame, model, seuil_distance=0.6):
    # trouver les emplacements des visages
    frame_face_locations = face_recognition.face_locations(frame)
    # Si aucun visage n'est trouvé dans l'image, retournez un résultat vide.
    if len(frame_face_locations) == 0:
        return []

    # Rechercher des encodages pour les visages dans l'image de test
    faces_encodings = face_recognition.face_encodings(frame, known_face_locations=frame_face_locations)

    # Utilisez le modèle KNN pour trouver les meilleures correspondances pour le visage de test
    closest_distances = model.kneighbors(faces_encodings, n_neighbors=1)
    matching = [closest_distances[0][i][0] <= seuil_distance for i in range(len(frame_face_locations))]

    # Prédire les classes et supprimer les classifications qui ne sont pas dans le seuil
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(model.predict(faces_encodings), frame_face_locations, matching)]


def main():    
    cap = cv2.VideoCapture(0)
    while True:
        # Prenez une seule image vidéo
        ret, frame = cap.read()

        # Afficher les résultats
        for  name,(top, right, bottom, left) in Prediction(frame, Model):
            # Dessinez une boîte autour du visage
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Dessinez une étiquette avec un nom sous le visage
            cv2.rectangle(frame, (left, top), (right, top - 25), (0, 0, 255), cv2.FILLED)  
            cv2.putText(frame, name, (left,top), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), 1)
    
        # Afficher l'image résultante
        cv2.imshow('Face recognition', frame)
        # Appuyez sur 'q' sur le clavier pour quitter!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Relâchez la poignée de la webcam
    cv2.destroyAllWindows()
    cap.release()
    file.close()
    
    
if __name__=='__main__':
     main()

