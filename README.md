

AgriSmart AI – AI-Based Smart Agriculture Irrigation Planner



\*\*GitHub Repository:\*\*  

https://github.com/dhyanu9747-eng/AI-Based-Smart-Agriculture-Irrigation-Planner



1. About the Project



AgriSmart AI is a Python-based application that uses machine learning, soil-moisture data, and weather information to support agricultural irrigation planning.



It provides soil-moisture predictions and irrigation-planning information through a simple interactive dashboard.



2. Main Features



\- Interactive dashboard

\- User registration and login

\- Farm and crop settings

\- Soil-moisture prediction for the next 15 minutes

\- Irrigation planning

\- Seven-day irrigation planning scenarios

\- Machine learning model comparison

\- Reports and insights

\- Data downloads and user feedback



3. Machine Learning Models



The project compares three approaches:



\- Linear Regression

\- Random Forest

\- Persistence Baseline



The models are evaluated using:



\- Mean Absolute Error (MAE)

\- Root Mean Squared Error (RMSE)

\- R-squared (R²)



Results are available in `models/model\_metrics.json`.



4. Dataset Information



The main model uses the EcoIrrigate soil-moisture dataset from Kolkata, India.



\- Study period: 5 January to 25 April 2025

\- Total records: 21,312

\- Measurement interval: 15 minutes

\- Farms: 2



The dataset documentation describes calibrated soil-moisture measurements and sensor-related information.



### Dataset Attribution

Dataset: EcoIrrigate: Multi-Sensor Soil Moisture Monitoring Dataset from Kolkata, India (Jan–Apr 2025).

Authors: Soham Saha, Subhojit Kar, Arijeet Ghosh, Sohan Bhattacharjee, Somes Sanyal, and Avik Kumar Das.

Published on Mendeley Data, Version 1, 2026.

DOI: https://doi.org/10.17632/8x4v7yy3ds.1

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).

Original dataset: https://data.mendeley.com/datasets/8x4v7yy3ds/1



5. Technologies Used



\- Python

\- Streamlit

\- Pandas

\- NumPy

\- Scikit-learn

\- Plotly

\- Joblib

\- Open-Meteo API



6. How to Run the Project



Step 1: Clone the repository



```bash

git clone https://github.com/dhyanu9747-eng/AI-Based-Smart-Agriculture-Irrigation-Planner.git

```



Step 2: Open the project folder



```bash

cd AI-Based-Smart-Agriculture-Irrigation-Planner

```



Step 3: Install the required libraries



```bash

python -m pip install -r requirements.txt

```



Step 4: Start the application



```bash

python -m streamlit run AI.py

```



7. Important Limitations



\- Model evaluation uses a chronological test split from the same dataset.

\- Results do not guarantee similar performance on other farms or soil types.

\- The model predicts soil moisture; it does not measure actual water savings or crop yield improvements.

\- The seven-day plan is a planning scenario, not a real weather forecast. It uses entered rainfall for day one and assumes zero rainfall for days two to seven.

\- The application does not connect to physical soil sensors, pumps, or automatic irrigation valves.

\- It is an educational decision-support prototype, not a replacement for expert agricultural advice.



8. Project Structure



\- `AI.py` – Main Streamlit application

\- `auth.py` – User authentication

\- `train\_model.py` – Model training and evaluation

\- `download\_data.py` – Downloads historical weather data and creates a separate estimated dataset

\- `feedback.py` – Feedback functionality

\- `data/` – Main dataset and documentation

\- `models/` – Trained model and evaluation results

\- `requirements.txt` – Required Python libraries



9. Project Purpose



This project demonstrates how machine learning and data analysis can support smarter irrigation planning and better understanding of soil-moisture conditions.



