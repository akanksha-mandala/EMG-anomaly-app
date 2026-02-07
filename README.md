EMG Anomaly Detection App

A Streamlit-based web application that analyzes EMG (electromyography) signals to detect abnormal muscle activity and provide clinical insights.

🌐 Live Demo

The app is deployed and accessible online:
Open EMG Anomaly App
https://emg-anomaly-app-agxuiwbcwbklm3umaqpk78.streamlit.app/

🧾 Features

Upload patient EMG CSV files easily.

Detect anomalies in muscle activation patterns automatically.

Visualize EMG data with abnormal spikes highlighted.

Navigate easily between Upload and Analysis pages.

💻 Requirements

Python 3.8+

Install required packages:

pip install -r requirements.txt

Example requirements.txt:

streamlit
pandas
numpy
matplotlib

📁 Project Structure
/baseline_emg/         # Baseline EMG CSV files
app.py                 # Main Streamlit application
requirements.txt       # Python dependencies
.gitignore
README.md

🧠 Notes
Ensure uploaded CSV files are numeric EMG data in correct format.
Clinical interpretations are for reference only, not a medical diagnosis.
The app works best with clean and correctly formatted EMG datasets.
