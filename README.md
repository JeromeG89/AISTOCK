# AISTOCK

This repository contains a stock prediction system that leverages historical data and machine learning techniques to generate reliable stock predictions. The predictions are generated using `main.py` and controlled via the `intoAPI.py` script.

---

## 🔧 Setup Instructions

Follow these steps to set up and run the AISTOCK system:

### 1. **Clone the Repository**
Clone this repository to your local system:
```bash
git clone https://github.com/JeromeG89/AISTOCK.git
cd AISTOCK
```

### 2. **Set Up Environment**
The application uses an `.env` file for storing sensitive configurations. Create an `.env` file in the root directory with the following structure:
```env
alpha_vantageAPI=YOUR_ALPHA_VANTAGE_API_KEY
```

Replace `YOUR_ALPHA_VANTAGE_API_KEY` with your Alpha Vantage API key.
Get your free API Key at https://www.alphavantage.co/support/#api-key

### 3. **Install Dependencies**
Install the required Python dependencies using `pip`:
```bash
pip install -r requirements.txt
```

### 4. **Set Up the Database (OPTIONAL)**
The `schema.sql` file contains the database schema required for the application. Initialize the database by running:
```bash
mysql -u root -p stock_logs < schema.sql
```

This creates the required tables in `database.db`.

### 5. **Run Predictions**
Run the `intoAPI.py` script to trigger predictions. This script internally calls `main.py` to perform the actual calculations:
```bash
python intoAPI.py
```
---

## 🚀 Running Details

### `intoAPI.py`
This script is the primary entry point to run the prediction system. It ensures that:
1. Necessary data is prepared.
2. Predictions are generated using the models in `main.py`.
3. Results are stored in the database.

### `main.py`
This script:
- Fetches stock data via the Alpha Vantage API.
- Prepares data for machine learning.
- Trains and applies predictive models.

---

## 📝 Notes
- Ensure you have a valid Alpha Vantage API key in your `.env` file.
- The system is designed for periodic execution, and it supports logging to monitor the performance of predictions.

---

## 📧 Support
For questions or issues, feel free to raise an issue in the repository or contact me directly.

---

## 📝 License
This project is licensed under the MIT License.

