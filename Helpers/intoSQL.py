import mysql.connector
import time
from datetime import datetime

def toMySQL(data):
    try:
        # print(f"Attempting to log data: {data}")
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Password12345!",
            database="stock_logs",
            connection_timeout=10
        )
        cursor = conn.cursor()
        query = '''
        INSERT INTO logs (ticker, log_date, price, n_weeks, prediction, output_date, confidence_train, confidence_test, confusion_matrix_score, auc_roc_score, predictor_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        '''
        cursor.execute(query, data)
        conn.commit()
        # print(f"Successfully logged: {data}")
    except Exception as e:
        print(f"Error logging data: {e}")
    finally:
        cursor.close()
        conn.close()

# log_to_mysql(('TESTabc', datetime.strptime("2024-12-27", "%Y-%m-%d").date(), 2, 'LONG', 1,1,1,1,1))
