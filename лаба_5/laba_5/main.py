"""
Консольное приложение для управления базой лиц и распознавания.

Зависимости (установите перед запуском):
    pip install opencv-python face_recognition numpy

Особенности:
- SQLite3 для хранения профилей (имя, эмбеддинг, путь к изображению).
- Добавление лиц из изображений: детекция, извлечение признаков, валидация.
- Распознавание по изображению или в реальном времени с веб-камеры.
"""

import base64
import os
import sqlite3
import threading
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import face_recognition
import numpy as np

DB_PATH = "faces.db"


def _ensure_db(path: str = DB_PATH) -> None:
    """Создаёт таблицу, если её ещё нет."""
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                encoding BLOB NOT NULL,
                image_path TEXT,
                created_at TEXT NOT NULL
            );
            """
        )


def _encode_face(image_path: str) -> np.ndarray:
    """Загружает изображение, находит лицо и возвращает 128-мерный вектор."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    image = face_recognition.load_image_file(image_path)
    boxes = face_recognition.face_locations(image)
    if not boxes:
        raise ValueError("Лицо не обнаружено на изображении")
    if len(boxes) > 1:
        raise ValueError("Обнаружено несколько лиц, выберите изображение с одним лицом")

    encodings = face_recognition.face_encodings(image, boxes)
    if not encodings:
        raise ValueError("Не удалось извлечь признаки лица")
    return encodings[0]


def _serialize_encoding(encoding: np.ndarray) -> bytes:
    """Преобразует np.ndarray в BLOB (base64 для читаемости)."""
    raw = encoding.astype(np.float32).tobytes()
    return base64.b64encode(raw)


def _deserialize_encoding(blob: bytes) -> np.ndarray:
    """Восстанавливает np.ndarray из BLOB."""
    raw = base64.b64decode(blob)
    return np.frombuffer(raw, dtype=np.float32)


def add_person(name: str, image_path: str, db_path: str = DB_PATH) -> int:
    """Добавляет профиль в БД и возвращает его id."""
    name = name.strip()
    if not name:
        raise ValueError("Имя не должно быть пустым")

    encoding = _encode_face(image_path)
    blob = _serialize_encoding(encoding)
    created_at = datetime.utcnow().isoformat()

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO persons (name, encoding, image_path, created_at) VALUES (?, ?, ?, ?)",
            (name, blob, image_path, created_at),
        )
        return cur.lastrowid


def list_persons(db_path: str = DB_PATH) -> List[Tuple[int, str, str]]:
    """Возвращает (id, имя, путь к изображению)."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT id, name, image_path FROM persons ORDER BY id")
        return list(cur.fetchall())


def _load_all_encodings(db_path: str = DB_PATH) -> Tuple[List[np.ndarray], List[str]]:
    """Загружает все эмбеддинги и имена."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT name, encoding FROM persons")
        rows = cur.fetchall()
    names, encodings = [], []
    for name, blob in rows:
        names.append(name)
        encodings.append(_deserialize_encoding(blob))
    return encodings, names


def recognize_from_image(
    image_path: str,
    db_path: str = DB_PATH,
    tolerance: float = 0.45,
) -> Optional[str]:
    """
    Распознаёт лицо на изображении и возвращает имя (или None).
    tolerance: чем меньше, тем строже сравнение.
    """
    encoding = _encode_face(image_path)
    encodings, names = _load_all_encodings(db_path)
    if not encodings:
        raise RuntimeError("База лиц пуста. Добавьте хотя бы один профиль.")

    distances = face_recognition.face_distance(encodings, encoding)
    if len(distances) == 0:
        return None
    best_idx = np.argmin(distances)
    if distances[best_idx] <= tolerance:
        return names[best_idx]
    return None


def delete_person(person_id: int, db_path: str = DB_PATH) -> bool:
    """Удаляет профиль из БД по id. Возвращает True если удалён."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        return cur.rowcount > 0


def recognize_from_camera(
    db_path: str = DB_PATH,
    tolerance: float = 0.45,
    camera_index: int = 0,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Распознавание в реальном времени с веб-камеры."""
    encodings, names = _load_all_encodings(db_path)
    if not encodings:
        raise RuntimeError("База лиц пуста. Добавьте хотя бы один профиль.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть веб-камеру")

    print("Нажмите 'q' для выхода")
    try:
        while True:
            if stop_event and stop_event.is_set():
                break
                
            ok, frame = cap.read()
            if not ok:
                print("Не удалось получить кадр с камеры")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)
            frame_encodings = face_recognition.face_encodings(rgb, locations)

            for (top, right, bottom, left), fe in zip(locations, frame_encodings):
                distances = face_recognition.face_distance(encodings, fe)
                name = "Неизвестно"
                if len(distances) > 0:
                    idx = np.argmin(distances)
                    if distances[idx] <= tolerance:
                        name = names[idx]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    name,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Распознавание", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def recognize_from_video(
    video_path: str,
    db_path: str = DB_PATH,
    tolerance: float = 0.45,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Распознавание лиц из видеофайла."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
    
    encodings, names = _load_all_encodings(db_path)
    if not encodings:
        raise RuntimeError("База лиц пуста. Добавьте хотя бы один профиль.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видеофайл: {video_path}")

    print("Нажмите 'q' для выхода")
    try:
        while True:
            if stop_event and stop_event.is_set():
                break
                
            ok, frame = cap.read()
            if not ok:
                print("Видео закончилось")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)
            frame_encodings = face_recognition.face_encodings(rgb, locations)

            for (top, right, bottom, left), fe in zip(locations, frame_encodings):
                distances = face_recognition.face_distance(encodings, fe)
                name = "Неизвестно"
                if len(distances) > 0:
                    idx = np.argmin(distances)
                    if distances[idx] <= tolerance:
                        name = names[idx]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    name,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Распознавание из видео", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def check_camera_available(camera_index: int = 0) -> bool:
    """Проверяет доступность веб-камеры."""
    cap = cv2.VideoCapture(camera_index)
    available = cap.isOpened()
    cap.release()
    return available


def _print_menu() -> None:
    print(
        """
Выберите действие:
1 — Добавить лицо в базу (из изображения)
2 — Показать список профилей
3 — Распознать лицо по изображению
4 — Распознавание в реальном времени с веб-камеры
0 — Выход
"""
    )


def main() -> None:
    _ensure_db()
    while True:
        _print_menu()
        choice = input("Ваш выбор: ").strip()

        try:
            if choice == "1":
                name = input("Введите имя: ").strip()
                path = input("Путь к изображению: ").strip()
                pid = add_person(name, path)
                print(f"Профиль добавлен, id={pid}")
            elif choice == "2":
                persons = list_persons()
                if not persons:
                    print("База пуста.")
                else:
                    for pid, name, path in persons:
                        print(f"[{pid}] {name} — {path}")
            elif choice == "3":
                path = input("Путь к изображению: ").strip()
                name = recognize_from_image(path)
                if name:
                    print(f"Распознано: {name}")
                else:
                    print("Совпадений не найдено")
            elif choice == "4":
                recognize_from_camera()
            elif choice == "0":
                print("Выход.")
                break
            else:
                print("Неверный пункт меню.")
        except Exception as exc:
            print(f"Ошибка: {exc}")


if __name__ == "__main__":
    main()
