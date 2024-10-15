import streamlit as st
import joblib
import pandas as pd
from urllib.parse import urlparse
import re
import tldextract
import math
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def extract_features(url):
    features = {}

    # Parse the URL
    parsed_url = urlparse(url)
    domain_info = tldextract.extract(url)

    # Length-based features
    features['url_length'] = len(url)
    features['hostname_length'] = len(parsed_url.netloc)
    features['path_length'] = len(parsed_url.path)

    # Count special characters in the URL
    features['special_char_count'] = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', url))

    # Hyphen count
    features['hyphen_count'] = url.count('-')

    # Subdomain count
    features['subdomain_count'] = parsed_url.netloc.count('.')

    # Check if URL uses IP address instead of domain
    ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'
    features['is_ip'] = 1 if re.search(ip_pattern, parsed_url.netloc) else 0

    # Calculate ratio of digits to total characters
    features['digit_ratio'] = sum(char.isdigit() for char in url) / len(url)

    # Check for HTTPS
    features['is_https'] = 1 if parsed_url.scheme == 'https' else 0

    # Number of parameters in query string
    features['param_count'] = len(parsed_url.query.split('&')) if parsed_url.query else 0

    # Presence of suspicious words in URL
    suspicious_words = ['verify', 'secure', 'login', 'signin', 'update', 'account', 'password']
    features['suspicious_word_count'] = sum([1 for word in suspicious_words if word in url.lower()])

    # Is the URL shortened?
    shortened_services = ['bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'is.gd', 'buff.ly', 'ow.ly']
    features['is_shortened'] = 1 if any(service in url for service in shortened_services) else 0

    # Check if the URL has uncommon TLD
    tld = domain_info.suffix
    suspicious_tlds = ['xyz', 'top', 'win', 'tk', 'cn', 'ml']
    features['uncommon_tld'] = 1 if tld in suspicious_tlds else 0

    # Check for port numbers in the URL
    features['contains_port'] = 1 if ':' in parsed_url.netloc else 0

    # Check if the URL contains redirection "//" after protocol
    features['has_redirection'] = 1 if '//' in parsed_url.path else 0

    # Check for suspicious file extensions
    suspicious_extensions = ['.exe', '.zip', '.rar', '.pdf']
    features['suspicious_extension'] = 1 if any(url.endswith(ext) for ext in suspicious_extensions) else 0

    # Check for the presence of anchor tag (#)
    features['anchor_tag'] = 1 if '#' in url else 0

    # Calculate entropy of the URL (measures randomness)
    def calculate_entropy(s):
        prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
        entropy = - sum([p * math.log2(p) for p in prob])
        return entropy

    features['url_entropy'] = calculate_entropy(url)

    # Presence of Unicode characters
    features['has_unicode'] = 1 if re.search(r'[^\x00-\x7F]+', url) else 0

    return features

# Load the saved pipeline
pipeline = joblib.load('phishing_detection_pipeline.joblib')


# Define the feature extraction function
def extract_features_from_url(url):
    features = extract_features(url)
    return pd.DataFrame([features])


st.title("Phishing URL Detection")

# User input for URL
user_url = st.text_input("Enter the URL to check:")

if st.button("Check URL"):
    if user_url:
        # Extract features from the URL
        features = extract_features_from_url(user_url)

        # Make prediction using the pipeline
        prediction = pipeline.predict(features)

        # Output the result
        if prediction[0] == "bad":
            st.error("Warning: This URL is likely a phishing attack!")
        else:
            st.success("This URL seems legitimate.")

        X_sample = pd.read_csv("value.csv")
        st.write("Random Forest Feature Importance:")
        rf_model = pipeline.named_steps['classifier']
        # Get feature importances from the RandomForest model
        feature_importances = rf_model.feature_importances_
        feature_names = X_sample.columns

        # Create a DataFrame for the feature importances
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

        # Sort by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title("Feature Importance in Random Forest")
        st.pyplot(plt)
    else:
        st.write("Please enter a URL to check.")
