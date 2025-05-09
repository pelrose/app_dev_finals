from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login as auth_login
from django.urls import reverse_lazy
from django.urls import reverse
from django.http import HttpResponseRedirect
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from .forms import PredictForm, CustomUserCreationForm
import os
import warnings
from sklearn.preprocessing import LabelEncoder
from django.contrib import messages

# Suppress scikit-learn warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Define the features used in training for spaceship model
SPACESHIP_FEATURES = [
    'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'
]
CATEGORICAL_FEATURES = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

# Load the model safely
try:
    model_path = os.path.join(os.path.dirname(__file__), 'ml_model', 'rospelspaceship.pkl')
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    print("Model loaded successfully")
    print("Model type:", type(model))
    print("Model parameters:", model.get_params())
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_spaceship_data(df, is_prediction=False):
    """Preprocess the data to match the spaceship model's expected format"""
    df_processed = df.copy()
    # Drop unused columns if present
    drop_cols = [col for col in ['PassengerId', 'Name', 'Cabin'] if col in df_processed.columns]
    df_processed = df_processed.drop(columns=drop_cols, errors='ignore')
    # Only keep relevant features
    if is_prediction:
        df_processed = df_processed[SPACESHIP_FEATURES]
    else:
        df_processed = df_processed[SPACESHIP_FEATURES + ['Transported']]
    # Fill missing values: categorical with mode, numeric with median
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    # Label encode categorical columns
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
    return df_processed

@login_required
def home(request):
    """Home page - accessible to all users"""
    try:
        # Load spaceship dataset
        df = pd.read_csv('train.csv')
        # Print debug information about the dataset
        print("\nDataset Info:")
        print("Total rows:", len(df))
        print("Features available:", df.columns.tolist())
        print("Missing values:", df.isnull().sum())
        # Create figures for multiple plots
        plt.figure(figsize=(15, 10))
        # Plot 1: Transported Rate Pie Chart
        plt.subplot(2, 2, 1)
        df['Transported'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Transported Rate')
        # Plot 2: Age Distribution by HomePlanet
        plt.subplot(2, 2, 2)
        sns.boxplot(x='HomePlanet', y='Age', data=df)
        plt.title('Age Distribution by HomePlanet')
        # Plot 3: RoomService Distribution
        plt.subplot(2, 2, 3)
        sns.histplot(data=df, x='RoomService', bins=30)
        plt.title('RoomService Distribution')
        # Plot 4: Transported by Destination
        plt.subplot(2, 2, 4)
        sns.countplot(data=df, x='Destination', hue='Transported')
        plt.title('Transported by Destination')
        # Adjust layout
        plt.tight_layout()
        # Save plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graph = base64.b64encode(image_png).decode('utf-8')
        # Calculate statistics
        stats = {
            'total_passengers': len(df),
            'transported_rate': round(df['Transported'].mean() * 100, 2),
            'avg_age': round(df['Age'].mean(), 2),
            'avg_roomservice': round(df['RoomService'].mean(), 2),
            # Transport rates by planet (percentage of passengers from each planet who were transported)
            'earth_rate': round(df[df['HomePlanet'] == 'Earth']['Transported'].mean() * 100, 2),
            'mars_rate': round(df[df['HomePlanet'] == 'Mars']['Transported'].mean() * 100, 2),
            'europa_rate': round(df[df['HomePlanet'] == 'Europa']['Transported'].mean() * 100, 2),
        }
        # Calculate model performance metrics if model is available
        if model is not None:
            try:
                df_processed = preprocess_spaceship_data(df, is_prediction=False)
                X = df_processed.drop('Transported', axis=1)
                y = df_processed['Transported']
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                y_pred = model.predict(X_test)
                accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
                # Confusion matrix
                plt.figure(figsize=(8, 6))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                buffer_cm = io.BytesIO()
                plt.savefig(buffer_cm, format='png')
                buffer_cm.seek(0)
                cm_image = base64.b64encode(buffer_cm.getvalue()).decode('utf-8')
                buffer_cm.close()
                return render(request, 'core/home.html', {
                    'stats': stats,
                    'graph': graph,
                    'cm_graph': cm_image,
                    'model_accuracy': accuracy
                })
            except Exception as e:
                print(f"Error calculating model metrics: {e}")
                return render(request, 'core/home.html', {
                    'stats': stats,
                    'graph': graph,
                    'error': 'Error calculating model performance metrics'
                })
        return render(request, 'core/home.html', {
            'stats': stats,
            'graph': graph
        })
    except Exception as e:
        print(f"Error in home: {e}")
        return render(request, 'core/error.html', {
            'message': f'Error loading dashboard: {str(e)}'
        })

# Register page (public)
def register(request):
    """Registration page - accessible to all users"""
    if request.user.is_authenticated:
        return redirect('home')
        
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            messages.success(request, 'Registration successful! Welcome to Titanic Survival Predictor.')
            return redirect('home')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CustomUserCreationForm()
    return render(request, 'core/register.html', {'form': form})

# Predict page (login required)
@login_required
def predict(request):
    """Prediction page - restricted to logged-in users"""
    if model is None:
        messages.error(request, 'Model not loaded. Please contact administrator.')
        return render(request, 'core/error.html', {
            'message': 'Model not loaded. Please contact administrator.'
        })
    if request.method == 'POST':
        form = PredictForm(request.POST)
        if form.is_valid():
            try:
                data = form.cleaned_data
                df = pd.DataFrame([data])
                # Preprocess for spaceship model
                df_processed = preprocess_spaceship_data(df, is_prediction=True)
                prediction = model.predict(df_processed)[0]
                return render(request, 'core/predict.html', {
                    'form': form,
                    'prediction': prediction,
                    'show_result': True
                })
            except Exception as e:
                print(f"Error making prediction: {e}")
                messages.error(request, f'Error making prediction: {str(e)}')
                return render(request, 'core/predict.html', {
                    'form': form,
                    'error': f'Error making prediction: {str(e)}'
                })
    else:
        form = PredictForm()
    return render(request, 'core/predict.html', {'form': form})

def landing(request):
    """Landing page - accessible to all users"""
    if request.user.is_authenticated:
        return redirect('home')
    return render(request, 'core/landing.html', {
        'title': 'Welcome to Titanic Survival Predictor',
        'description': 'Predict survival chances on the Titanic using machine learning.'
    })
