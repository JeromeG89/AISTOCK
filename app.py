import mysql.connector
from flask import Flask, jsonify


app = Flask(__name__)

@app.route('/api/logs', methods=['GET'])
def get_logs():
    try:
        # Connect to your MySQL database
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Password12345!",
            database="stock_logs"
        )
        cursor = conn.cursor(dictionary=True)
        print('connected')
        # Fetch data from your database
        cursor.execute("SELECT ticker, log_date, price, n_weeks, prediction, output_date, confidence_train, confidence_test, confusion_matrix_score, auc_roc_score, predictor_version FROM logs")
        logs = cursor.fetchall()

        # Close the connection
        conn.close()

        # Return the data as JSON
        return jsonify(logs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
