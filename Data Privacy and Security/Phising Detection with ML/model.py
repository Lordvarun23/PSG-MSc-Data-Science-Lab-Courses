import re
from urllib.parse import urlparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from joblib import dump
import tldextract
import math

# Feature extraction from a URL
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


def url_to_features(url):
    return pd.DataFrame([extract_features(url)])
## EDA
# Load the dataset
df = pd.read_csv('C:\\Projects\\College\\Data Privacy and Security\\Package\\phishing_site_urls.csv')
plt.bar(df['Label'].value_counts().index,df['Label'].value_counts().values)
plt.show()
from sklearn.utils import resample

# Assuming 'label' is the target column where:
# 0 = Legitimate, 1 = Phishing

# Separate majority and minority classes
df_majority = df[df['Label'] == "good"]  # Legitimate
df_minority = df[df['Label'] == "bad"]  # Phishing

# Downsample majority class
df_majority_downsampled = resample(df_majority,
                                   replace=False,    # sample without replacement
                                   n_samples=len(df_minority),  # Match minority class size
                                   random_state=42)  # For reproducibility

# Combine downsampled majority class with minority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

# Shuffle the dataset to mix the rows
df = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the new class balance
print(df_downsampled['Label'].value_counts())
plt.bar(df_downsampled['Label'].value_counts().index,df_downsampled['Label'].value_counts().values)
plt.show()
'''
# Apply feature extraction to all URLs
url_features = df['URL'].apply(lambda x: pd.Series(extract_features(x)))

# Combine the features with the original DataFrame
df = pd.concat([df, url_features], axis=1)

# Drop the original 'url' column if needed
df.drop(columns=['URL'], inplace=True)

# Split into features and labels
X = df.drop('Label', axis=1)
y = df['Label']

# Now, you can handle missing values, if any
X.fillna(X.median(), inplace=True)
X.to_csv("value.csv",index=False)
# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit the pipeline
pipeline.fit(X, y)

print(classification_report(y_test,pipeline.predict(X_test)))

# Save the pipeline
dump(pipeline, 'phishing_detection_pipeline.joblib')
'''