# EMG Signal Analysis System

A Streamlit application designed to analyze EMG (electromyography) signals, detect anomalies, and provide clinical interpretations for doctors and researchers.

## Overview

This system allows doctors to upload patient EMG data in CSV format. It visualizes the EMG signal, identifies abnormal spikes, and generates a clinical summary and interpretation.

## Key Features

- Upload EMG data (CSV format)  
- Interactive EMG signal visualization with anomaly markers  
- Automatic detection of abnormal muscle activity  
- Clinical summary including patient ID, muscle type, and detected anomalies  
- PDF report generation for record-keeping  
- User-friendly interface for doctors  

## Tech Stack

- **Interface:** Streamlit  
- **Data Processing:** Python, Pandas, NumPy  
- **Visualization:** Matplotlib, Plotly  
- **Reporting:** FPDF  

## How It Works

1. Doctor uploads patient EMG CSV data.  
2. The system plots the EMG signal and highlights anomalies.  
3. Detected anomalies are evaluated against normal muscle activity patterns.  
4. A clinical summary and interpretation are displayed.  
5. Optional: generate a PDF report with all details.

## Project Status

This project is a functional prototype and can be extended with:

- Real-time EMG data streaming  
- Advanced anomaly detection using AI/ML models  
- Integration with hospital management systems  
- Deployment on cloud platforms  

## Author

**Akanksha Mandala**
