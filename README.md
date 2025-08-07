# Crop Recommendation and Yield Prediction using Machine Learning

## Project Overview
This project provides an AI-powered system to help farmers:
- **Recommend the best crop** to grow based on soil and weather conditions.
- **Predict the expected yield** (tons/hectare) for the recommended crop using historical data.

## Tech Stack
- **Python 3.8+**
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, joblib
- **Jupyter Notebook** for EDA and model training
- **Streamlit** for the web app UI

## Data Sources
- [Crop Recommendation Dataset (Kaggle)](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- [Crop Yield Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/abhinand05/crop-yield-prediction-dataset)

### Data Structure
- **crop_recommendation.csv:** N, P, K, temperature, humidity, ph, rainfall, label (crop name)
- **weather_yield_data.csv:** state, district, crop, year, season, rainfall, temperature, yield (tons/hectare)

## Project Structure
```
zero_hunger_ai_project/
│
├── data/
│   ├── crop_recommendation.csv
│   └── weather_yield_data.csv
│
├── notebooks/
│   └── EDA_and_Model_Training.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── crop_recommendation_model.py
│   ├── yield_prediction_model.py
│   └── utils.py
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Workflow
1. **Data Loading & Cleaning:**
   - Load both datasets, handle missing values, encode categorical variables.
2. **Exploratory Data Analysis (EDA):**
   - Visualize crop class distribution, feature correlations, and average yields.
3. **Model Training:**
   - **Crop Recommendation:** RandomForestClassifier predicts the best crop.
   - **Yield Prediction:** RandomForestRegressor predicts expected yield.
4. **Model Evaluation:**
   - Classification accuracy, classification report, and regression R² score.
5. **Model Saving:**
   - Trained models are saved with joblib for use in the Streamlit app.
6. **Deployment:**
   - Streamlit app provides an interactive UI for predictions.

## How to Run
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Streamlit app:**
   ```bash
   streamlit run app/streamlit_app.py
   ```
3. **Jupyter Notebook:**
   - Open `notebooks/EDA_and_Model_Training.ipynb` for EDA and model training.

## Example Outputs
- **Crop Recommendation Model Accuracy:** 1.00 (on sample data)
- **Classification Report:**
  - Shows precision, recall, f1-score for each crop class.
- **Yield Prediction Model R² Score:** 0.63
- **Mean Squared Error:** 1.06

## Usage in Streamlit App
- **Crop Recommendation:**
  - Input: N, P, K, temperature, humidity, pH, rainfall
  - Output: Best crop to grow
- **Yield Prediction:**
  - Input: crop name, season, year, rainfall, temperature
  - Output: Expected yield (tons/hectare)

## Notes
- For best results, use real and larger datasets.
- Models and workflow can be improved with more data and feature engineering.

## License
This project is for educational and research purposes.
