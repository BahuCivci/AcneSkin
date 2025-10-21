from pathlib import Path
import os
import sqlite3

class StorageService:
    def __init__(self, db_path='storage.db', upload_folder='uploads'):
        self.db_path = db_path
        self.upload_folder = upload_folder
        self.create_upload_folder()
        self.create_database()

    def create_upload_folder(self):
        Path(self.upload_folder).mkdir(parents=True, exist_ok=True)

    def create_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                analysis_results TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def save_image(self, image_file):
        image_path = os.path.join(self.upload_folder, image_file.name)
        with open(image_path, 'wb') as f:
            f.write(image_file.getbuffer())
        return image_path

    def store_analysis_results(self, image_path, results):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_data (image_path, analysis_results)
            VALUES (?, ?)
        ''', (image_path, results))
        conn.commit()
        conn.close()

    def get_analysis_results(self, image_path):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT analysis_results FROM user_data WHERE image_path = ?
        ''', (image_path,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None