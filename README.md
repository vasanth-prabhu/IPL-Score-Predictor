# IPL-Score-Predictor

Predict the total runs in an IPL match using Machine Learning and Deep Learning!

## Overview

This project provides interactive tools and models to predict the **total runs** scored in an IPL cricket match, based on real match conditions and player/team information. It includes:

- A Streamlit web app for easy predictions.
- Jupyter notebooks for both Machine Learning and Deep Learning approaches.
- Pre-trained models and encoders for quick deployment.

## Features

- Predict total runs using match context (venue, teams, batsmen, bowler, current score, etc.).
- Interactive web app built with Streamlit.
- Notebooks for data exploration, model training, and evaluation.
- Both ML (Random Forest, Gradient Boosting, Linear Regression) and DL (Keras/TensorFlow) models.
- Ready-to-use trained models and preprocessing tools.

## Project Structure

```
.
├── ipl_data.csv                        # IPL ball-by-ball dataset
├── Score_Predictor_App.py              # Streamlit web app
├── IPL_Score_Prediction_using_Machine_Learning.ipynb
├── IPL_Score_Predictor_using_Deep_Learning.ipynb
├── Best Model using DL/
│   ├── best_model.keras                # Trained deep learning model
│   ├── label_encoders.pkl              # Encoders for categorical features
│   └── scaler.pkl                      # Scaler for feature normalization
├── LICENSE
└── README.md
```

## Getting Started

### 1. Clone the repository

```sh
git clone https://github.com/yourusername/IPL-Score-Predictor.git
cd IPL-Score-Predictor
```

### 2. Install dependencies

```sh
pip install -r requirements.txt
```
Or manually install:
```sh
pip install streamlit pandas numpy scikit-learn keras tensorflow joblib
```
### 3. Explore the Notebooks

Open the Jupyter notebooks for detailed data analysis, model training, and evaluation:

- [`IPL_Score_Prediction_using_Machine_Learning.ipynb`](IPL_Score_Prediction_using_Machine_Learning.ipynb)
- [`IPL_Score_Predictor_using_Deep_Learning.ipynb`](IPL_Score_Predictor_using_Deep_Learning.ipynb)

### 4. Run the Streamlit App
- Run the [`IPL_Score_Prediction_using_Machine_Learning.ipynb`](IPL_Score_Prediction_using_Machine_Learning.ipynb) notebook and after running the last cell save the best_ML_model_pipeline.joblib file in the cloned repository.
- Then run the  app by the following command:
```sh
streamlit run Score_Predictor_App.py
```

- You can also run the app with Deep Learning model by just using the best obtained model and also by replacing the encoders and scaler.


## Data

The dataset [`ipl_data.csv`](ipl_data.csv) contains ball-by-ball IPL match data, including:

- Venue, Batting/Bowling teams
- Batsman, Bowler
- Current runs, wickets, overs
- Last 5 overs stats
- Striker/non-striker runs
- Total runs in the innings

## Model Details

- **Preprocessing:** Label encoding for categorical features, MinMax scaling for numerical features
- **ML Models:** Random Forest, Gradient Boosting, Linear Regression (scikit-learn)
- **DL Model:** Keras Sequential model with dense layers and dropout

## Results

- Models are evaluated using Mean Squared Error, Mean Absolute Error, and R² score.
- The best model is selected based on validation performance.
- Among all the models trained and tested, the Random forest model showed a R² score of 0.93 which was the highest.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- Built with [Streamlit](https://streamlit.io/), [scikit-learn](https://scikit-learn.org/), [Keras](https://keras.io/), and [TensorFlow](https://www.tensorflow.org/)

---

**Enjoy predicting IPL scores!**