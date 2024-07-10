import streamlit as st
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from urllib.parse import urlparse

# Load dataset
st.title('Phishing Website Detection')
st.write('This application detects whether a website is a phishing website or not using various machine learning models.')

# Fetch dataset
phishing_websites = fetch_ucirepo(id=327)

# Extract feature names and target column
feature_names = phishing_websites.variables[phishing_websites.variables['role'] == 'Feature']['name'].tolist()
target_name = phishing_websites.variables[phishing_websites.variables['role'] == 'Target']['name'].iloc[0]

# Ensure consistent feature names
feature_names = [feature.lower() for feature in feature_names]

# Convert features and targets to DataFrame and Series respectively
X = pd.DataFrame(phishing_websites.data.features, columns=feature_names)
y = phishing_websites.data.targets[target_name]  # Correctly extract the target column as a Series

# Map target values from [-1, 1] to [0, 1]
y_mapped = y.map({-1: 0, 1: 1})

# Split the data using the mapped target values
X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, random_state=42)

# Sidebar for model selection
st.sidebar.title('Model Selection')
model_choice = st.sidebar.selectbox('Choose Model', ('Gradient Boosting', 'XGBoost', 'LightGBM', 'Random Forest', 'Voting Classifier'))

# Function to train and evaluate model
def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.subheader(f'{model_name} Model Results')
    st.write('Classification Report:')
    st.text(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Accuracy: {accuracy:.2f}')

# Function for hyperparameter tuning
def tune_hyperparameters(model, param_grid):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Hyperparameters for tuning
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.01],
    'max_depth': [3, 4, 5]
}

xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.01],
    'max_depth': [3, 4, 5]
}

lgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.01],
    'num_leaves': [31, 40, 50]
}

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Train and evaluate the selected model
if model_choice == 'Gradient Boosting':
    model = GradientBoostingClassifier()
    model = tune_hyperparameters(model, gb_param_grid)
    train_and_evaluate(model, 'Gradient Boosting')
elif model_choice == 'XGBoost':
    model = xgb.XGBClassifier()
    model = tune_hyperparameters(model, xgb_param_grid)
    train_and_evaluate(model, 'XGBoost')
elif model_choice == 'LightGBM':
    model = lgb.LGBMClassifier()
    model = tune_hyperparameters(model, lgb_param_grid)
    train_and_evaluate(model, 'LightGBM')
elif model_choice == 'Random Forest':
    model = RandomForestClassifier()
    model = tune_hyperparameters(model, rf_param_grid)
    train_and_evaluate(model, 'Random Forest')
elif model_choice == 'Voting Classifier':
    model1 = GradientBoostingClassifier(**tune_hyperparameters(GradientBoostingClassifier(), gb_param_grid).get_params())
    model2 = xgb.XGBClassifier(**tune_hyperparameters(xgb.XGBClassifier(), xgb_param_grid).get_params())
    model3 = lgb.LGBMClassifier(**tune_hyperparameters(lgb.LGBMClassifier(), lgb_param_grid).get_params())
    model = VotingClassifier(estimators=[('gb', model1), ('xgb', model2), ('lgb', model3)], voting='soft')
    train_and_evaluate(model, 'Voting Classifier')

st.write('---')
st.write('Created by Ashutosh')

# Create input fields for user data
st.sidebar.title('Input Data')
user_data = {}
for feature in feature_names:
    user_data[feature] = st.sidebar.number_input(feature, min_value=0.0, max_value=1.0, step=0.01)

user_input = pd.DataFrame(user_data, index=[0])

if st.sidebar.button('Predict'):
    model.fit(X, y_mapped)
    prediction = model.predict(user_input)
    st.write(f'Prediction: {"Phishing Website" if prediction[0] == 1 else "Legitimate Website"}')

# Add URL input section
st.sidebar.title('URL Prediction')
url_input = st.sidebar.text_input('Enter a website URL')

def extract_features_from_url(url):
    parsed_url = urlparse(url)
    features = {
        'having_ip_address': int(parsed_url.hostname.replace('.', '').isdigit()),
        'url_length': len(url),
        'shortining_service': int(any(short in url for short in ['bit.ly', 'tinyurl.com'])),
        'having_at_symbol': int('@' in url),
        'double_slash_redirecting': int('//' in parsed_url.path),
        'prefix_suffix': int('-' in parsed_url.netloc),
        'having_sub_domain': len(parsed_url.netloc.split('.')) - 2,
        'sslfinal_state': int(parsed_url.scheme == 'https'),
        'domain_registration_length': 1,  # Placeholder, need actual data
        'favicon': 1,  # Placeholder, need actual data
        'port': int(parsed_url.port is not None and parsed_url.port != 80),
        'https_token': int('https' in parsed_url.netloc),
        'request_url': 1,  # Placeholder, need actual data
        'url_of_anchor': 1,  # Placeholder, need actual data
        'links_in_tags': 1,  # Placeholder, need actual data
        'sfh': 1,  # Placeholder, need actual data
        'submitting_to_email': 1,  # Placeholder, need actual data
        'abnormal_url': 1,  # Placeholder, need actual data
        'redirect': 0,  # Assuming no redirection for simplification
        'on_mouseover': 0,  # Assuming no mouseover for simplification
        'rightclick': 0,  # Assuming no rightclick for simplification
        'popupwindow': 0,  # Assuming no popup for simplification
        'iframe': 0,  # Assuming no iframe for simplification
        'age_of_domain': 1,  # Placeholder, need actual data
        'dnsrecord': 1,  # Placeholder, need actual data
        'web_traffic': 1,  # Placeholder, need actual data
        'page_rank': 1,  # Placeholder, need actual data
        'google_index': 1,  # Placeholder, need actual data
        'links_pointing_to_page': 1,  # Placeholder, need actual data
        'statistical_report': 1,  # Placeholder, need actual data
    }
    return pd.DataFrame(features, index=[0])

if st.sidebar.button('Predict URL'):
    model.fit(X, y_mapped)
    url_features = extract_features_from_url(url_input)
    url_features.columns = [col.lower() for col in url_features.columns]  # Ensure feature names are in lowercase
    X.columns = [col.lower() for col in X.columns]  # Ensure training feature names are in lowercase
    url_prediction = model.predict(url_features)
    st.write(f'URL Prediction: {"Phishing Website" if url_prediction[0] == 0 else "Legitimate Website"}')

st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
        color: #000;
    }
    .sidebar .sidebar-content {
        background: #c9cdd4;
    }
    </style>
    """,
    unsafe_allow_html=True
)
