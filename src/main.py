import cv2
import numpy as np
import random

# Lista losowych nazw zwierząt
animal_names = ["bear", "cat", "dog", "elephant", "fox", "giraffe", "hippo", "jaguar", "kangaroo", "lion", "monkey", "owl", "penguin", "quokka", "rhino", "snake", "tiger", "unicorn", "vulture", "wolf", "x-ray fish", "yeti", "zebra"]

# Uruchomienie kamery internetowej
cap = cv2.VideoCapture(0)

# Detekcja twarzy za pomocą Haarcascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Lista przechowująca pary (rect, name), gdzie rect to informacje o twarzy, a name to przypisane imię
faces_with_names = []

# Liczniki do śledzenia uśmiechów i wszystkich twarzy
smile_count = 0
total_faces = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konwersja kolorów do skali szarości dla detekcji twarzy
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detekcja twarzy
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Wyciąganie obszaru z twarzą
        face_roi = frame[y:y + h, x:x + w]

        # Określanie, czy twarz jest uśmiechnięta
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.5, minNeighbors=20, minSize=(25, 25))

        total_faces += 1  # Zwiększenie liczby wszystkich twarzy

        if len(smiles) > 0:
            smile_text = "Smiling"
            smile_count += 1  # Zwiększenie liczby uśmiechów
        else:
            smile_text = "Not Smiling"

        # Sprawdzanie, czy twarz jest już w liście
        is_known_face = False
        for rect, name in faces_with_names:
            if x <= rect[0] <= (x + w) and y <= rect[1] <= (y + h):
                is_known_face = True
                # Wyświetlanie imienia obok twarzy
                cv2.putText(frame, name, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                break

        # Jeśli twarz nie jest w liście, przypisz losową nazwę zwierzęcia
        if not is_known_face:
            name = random.choice(animal_names)
            faces_with_names.append(((x, y, w, h), name))

        # Rysowanie prostokąta wokół twarzy
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Obliczanie procentowego udziału uśmiechu
        if total_faces > 0:
            smile_percentage = (smile_count / total_faces) * 100

            # Wyświetlanie informacji o uśmiechu i % uśmiechu obok twarzy
            cv2.putText(frame, f"{smile_text} - {smile_percentage:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Wyświetlanie obrazu
    cv2.imshow('Live Facial Recognition', frame)

    # Przerwanie pętli po naciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zamknięcie kamery i okien
cap.release()
cv2.destroyAllWindows()