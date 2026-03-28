import psycopg2
import os

def get_connection():
    return psycopg2.connect(
        host="localhost",
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


def save_prediction(image_url, predicted, correct, confidence):
    conn = get_connection()
    cur = conn.cursor()

    is_correct = predicted == correct

    cur.execute("""
        INSERT INTO predictions (image_url, predicted_label, correct_label, confidence, is_correct)
        VALUES (%s, %s, %s, %s, %s)
    """, (image_url, predicted, correct, confidence, is_correct))

    conn.commit()
    cur.close()
    conn.close()


def get_all_predictions():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM predictions")
    data = cur.fetchall()

    cur.close()
    conn.close()

    return data