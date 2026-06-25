# AI-Driven Intrusion Detection System (IDS)

A powerful cybersecurity project that uses machine learning to detect network intrusions. This system monitors network traffic and automatically identifies suspicious or malicious activities using an ensemble of ML models.

## Features

- **Real-time Detection:** Analyze single network traffic records.
- **Batch Analysis:** Process large volumes of traffic data from CSV files.
- **Interactive Dashboard:** Visualize traffic patterns, attack types, and protocol distributions.
- **Ensemble Learning:** Uses Random Forest, Neural Networks, and Logistic Regression for high accuracy.
- **Explainable AI:** Provides confidence scores and reasoning for detections.

## Technology Stack

- **Frontend:** Streamlit
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly
- **Dataset:** NSL-KDD

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Local Installation & Setup

1. **Clone or Download the Project:**
   ```bash
   git clone <repository-url>
   cd ai_ids_project
   ```

2. **Install Dependencies:**
   ```bash
   pip install streamlit pandas scikit-learn joblib matplotlib seaborn plotly
   ```

3. **Train the Models:**
   (Note: Ensure `train.csv` and `test.csv` are in the project directory)
   ```bash
   python train_models.py
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

5. **Access the App:**
   Open your browser and go to `http://localhost:8501`

## Project Structure

- `app.py`: Main Streamlit application.
- `train_models.py`: Script to train and save ML models.
- `models/`: Directory containing saved models and encoders.
- `train.csv`: Training dataset (NSL-KDD).
- `test.csv`: Testing dataset (NSL-KDD).
- `README.md`: Project documentation.

## Dataset Information

The project uses the **NSL-KDD dataset**, a refined version of the KDD'99 dataset. It includes 41 features describing network connections and labels them as 'normal' or various types of 'attacks' (DoS, Probe, R2L, U2R).

## Key Benefits

- **Detects Zero-Day Attacks:** Identifies unknown threats based on patterns.
- **Reduces False Alerts:** Ensemble voting improves reliability.
- **Adaptive:** Learns from historical data to stay ahead of attackers.

---
**Disclaimer:** This project is for educational purposes. For production use, ensure proper security auditing and compliance.
