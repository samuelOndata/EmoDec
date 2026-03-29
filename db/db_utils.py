import psycopg2
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    try:
        st.write(f"Hôte détecté : '{os.getenv('DB_HOST')}'")

        return psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "6543"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        
    except Exception as e:
        st.error(f"Erreur de connexion postgres : {e}")


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


# @st.cache_data(ttl=600)
def get_all_predictions():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM predictions")
    data = cur.fetchall()

    cur.close()
    conn.close()

    return data