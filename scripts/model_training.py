import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten, Input, SimpleRNN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import mlflow
import mlflow.sklearn
import mlflow.keras
import time


class ModelTraining:
    def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.performance_metrics = {}
        self.y_probs = {}

    def add_models(self):
        """Initialize models and store them in the self.models dictionary."""
        self.models['Logistic Regression'] = LogisticRegression()
        self.models['Decision Tree'] = DecisionTreeClassifier()
        self.models['Random Forest'] = RandomForestClassifier()
        self.models['Gradient Boosting'] = GradientBoostingClassifier()
        self.models['MLP'] = MLPClassifier()
        self.models['LSTM'] = self.build_lstm_model()
        self.models['CNN'] = self.build_cnn_model()
        self.models['RNN'] = self.build_rnn_model()

    def build_lstm_model(self):
        model = Sequential([
            Input(shape=(self.X_train.shape[1], 1)),
            LSTM(50),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def build_rnn_model(self):
        model = Sequential([
            Input(shape=(self.X_train.shape[1], 1)),
            SimpleRNN(50),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def build_cnn_model(self):
        model = Sequential([
            Input(shape=(self.X_train.shape[1], 1)),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def plot_training_history(self, history, model_name="Model"):
        """
        Plots the training and validation loss and accuracy over epochs.
        
        Parameters:
        - history: History object from model.fit(), containing training and validation metrics.
        - model_name: Optional; Name of the model being trained, used in the plot title.
        """
        has_val = 'val_loss' in history.history and 'val_accuracy' in history.history

        plt.figure(figsize=(14, 6))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        if has_val:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy plot (if available)
        if 'accuracy' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            if has_val:
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'{model_name} Accuracy over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def hyperparameter_tuning(self):
        """Perform GridSearchCV for hyperparameter tuning of sklearn models."""
        param_grids = {
            'Logistic Regression': {'classifier__C': [0.1, 1, 10]},
            'Decision Tree': {'classifier__max_depth': [None, 5, 10]},
            'Random Forest': {'classifier__n_estimators': [50, 100]},
            'Gradient Boosting': {'classifier__learning_rate': [0.01, 0.1]},
            'MLP': {'classifier__hidden_layer_sizes': [(50,), (100,)]}
        }

        best_models = {}
        for name, model in self.models.items():
            if name in ['LSTM', 'CNN', 'RNN']:
                continue  # Skip neural networks for hyperparameter tuning

            print(f"Tuning {name}...")
            pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', model)])
            search = GridSearchCV(pipeline, param_grid=param_grids[name], cv=3, scoring='accuracy', n_jobs=-1)
            search.fit(self.X_train, self.y_train)
            best_models[name] = search.best_estimator_
            print(f"{name} best parameters: {search.best_params_}")

        self.models.update(best_models)

    def train_and_evaluate(self):
        """Train models, evaluate their performance, and return the best model."""
        self.add_models()  # Ensure models are initialized
        self.hyperparameter_tuning()

        best_model = None
        best_model_name = None
        best_score = float('-inf')

        for name, model in self.models.items():
            with mlflow.start_run(run_name=name):
                start_time = time.time()

                if name in ['LSTM', 'CNN', 'RNN']:
                    # Reshape input for neural networks
                    X_train_reshaped = self.X_train.values.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
                    X_test_reshaped = self.X_test.values.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)

                    history = model.fit(
                        X_train_reshaped, self.y_train,
                        validation_data=(X_test_reshaped, self.y_test),
                        epochs=5, batch_size=32, verbose=0
                    )
                    y_prob = model.predict(X_test_reshaped).flatten()
                    y_pred = (y_prob > 0.5).astype("int32")
                    # Plot training history
                    self.plot_training_history(history, model_name=name)
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                    y_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

                # Compute performance metrics
                acc = accuracy_score(self.y_test, y_pred)
                prec = precision_score(self.y_test, y_pred)
                rec = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                roc_auc = roc_auc_score(self.y_test, y_prob)

                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc_auc)

                # Store performance metrics
                self.performance_metrics[name] = {
                    'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'roc_auc': roc_auc
                }
                self.y_probs[name] = y_prob

                # Log metrics to MLflow
                mlflow.log_metrics({
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1,
                    "roc_auc": roc_auc
                })

                # Save the model using MLflow
                if name in ['LSTM', 'CNN', 'RNN']:
                    mlflow.keras.log_model(model, f"{name.lower()}_model")
                else:
                    mlflow.sklearn.log_model(model, f"{name.lower()}_model")

                print(f"{name} model trained and logged with MLflow")
                # Compute a weighted score
                weighted_score = 0.4 * acc + 0.3 * f1 + 0.3 * roc_auc

                # Track the best model based on accuracy
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_model = model
                    best_model_name = name

        return best_model, best_model_name  # Ensure this is returned

    def save_best_models(self, best_model, best_model_name, dataset_name):
        """Save the best performing model."""
        if best_model is None or best_model_name is None:
            print("No best model found to save.")
            return

        sanitized_name = best_model_name.replace(' ', '_').lower()
        joblib.dump(best_model, f"../api/{sanitized_name}_{dataset_name}_best_model.pkl")
        print(f"{best_model_name} best model saved.")

    def get_results(self):
        """Return the performance metrics and predicted probabilities."""
        return self.performance_metrics, self.y_probs
