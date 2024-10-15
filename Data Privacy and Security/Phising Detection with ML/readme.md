# PhishGuard: URL-Based Phishing Detection System

PhishGuard is a robust machine learning application designed to detect phishing URLs using advanced feature extraction techniques. This project aims to provide users with a reliable method to assess the safety of URLs by leveraging various characteristics that are indicative of phishing attempts.

## Table of Contents

- [Features](#features)
- [Usage](#usage)
- [Model Development](#model-development)
- [Results](#results)
- [License](#license)
- [Contributing](#contributing)

## Features

- **Advanced Feature Extraction**: Extracts multiple features from URLs, including length, special characters, subdomains, and suspicious keywords.
- **Machine Learning Models**: Utilizes Logistic Regression, Random Forest, and XGBoost for classification.
- **Model Evaluation**: Implements accuracy, precision, recall, F1-score, and ROC-AUC for performance metrics.
- **Interactive UI**: Built with Streamlit, allowing users to input URLs and receive immediate feedback on phishing risks.
- **Interpretability**: Includes LIME plots for model interpretability, showcasing how features contribute to predictions.

## Usage

- Start the Streamlit application:
```bash
streamlit run app.py
```
- Open your web browser and go to http://localhost:8501.
- Enter the URL you want to check in the input box and click the "Check" button.

## Model Development
The model development process includes:

- Data collection from reputable datasets.
- Extensive Exploratory Data Analysis (EDA) to understand class balance and feature distributions.
- Preprocessing techniques, including downsampling of the dataset for balanced classes.
- Implementation of feature extraction functions to convert URLs into usable features for model training.
- Training and evaluation of multiple machine learning models to determine the best performer.

## Results
PhishGuard achieved promising results in terms of accuracy and interpretability. The LIME plots allow users to understand why a particular URL was classified as phishing or legitimate, enhancing trust in the model's predictions.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request to help improve PhishGuard.
