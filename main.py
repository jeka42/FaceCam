#Код демонстрации запуска распознавания лиц на видео в реальном времени
import cv2
import face_recognition
import numpy as np

# Получаем ссылку на видеофайл, либо с веб камеры, то изменить ('sample_video.mp4') на (0)
video_capture = cv2.VideoCapture('sample_video.mp4')

# Ссылаемся на образец изображения в базе и учим его распознавать
# Под каждое лицо пишем свой код с именем картинки
# image = face_recognition.load_image_file('sample_image.png')
# face_encoding = face_recognition.face_encodings(image) [0]
image = face_recognition.load_image_file('sample_image.png')
face_encoding = face_recognition.face_encodings(image) [0]

# Создаем массив кодировки лица и имени
# Под каждое лицо добавляем код с другими данными
known_face_encodings = [face_encoding]
known_face_names = ['Chuwi']

# Инициализация некоторых переменных
face_locations = []
face_encoding = []
face_names = []
process_this_frame = True

while True:
    # Берем один кадр из видео
    ret, frame = video_capture.read()
    # Изменяем размер кадра видео до 1/4 размера для более быстрой обработки распознавания
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Преобразуем цвет BGR в цвет RGB который использует Face_recognition
    rgb_small_frame = small_frame[:, :, ::-1]

    # Обрабатываем только каждый второй кадр видео, для экономии времени
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encoding = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encoding:
            # Смотрим совпадает ли лицо с известным лицом в базе
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = 'Unknown'

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Рисуем рамку вокруг лица
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Рисуем этикетку с именем под рамкой
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left +6, bottom -6), font, 1.0, (255, 255, 255), 1)

    # Показываем получившееся изображение
    cv2.imshow('Video', frame)

    # Нажмите 'q' чтобы выйти из просмотра
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Запуск кода
video_capture.release()
cv2.destroyAllWindows()