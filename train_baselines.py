import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import os

def train_baselines(data_dir, model_dir):
    # Load data
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    X_train = train_df['text']
    y_train = train_df['label']
    X_test = test_df['text']
    y_test = test_df['label']
    
    # 1. Majority Class Baseline
    majority_class = y_train.mode()[0]
    y_pred_majority = [majority_class] * len(y_test)
    
    majority_acc = accuracy_score(y_test, y_pred_majority)
    majority_f1 = f1_score(y_test, y_pred_majority, average='macro')
    
    print("--- Majority Class Baseline ---")
    print(f"Accuracy: {majority_acc:.4f}")
    print(f"Macro F1: {majority_f1:.4f}")
    
    # 2. TF-IDF + Logistic Regression
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Simple cross-validation to show consistency (optional but good practice)
    cv_scores = []
    for fold in range(5):
        fold_train = train_df[train_df['fold'] != fold]
        fold_val = train_df[train_df['fold'] == fold]
        
        X_f_train = vectorizer.transform(fold_train['text'])
        y_f_train = fold_train['label']
        X_f_val = vectorizer.transform(fold_val['text'])
        y_f_val = fold_val['label']
        
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_f_train, y_f_train)
        fold_acc = clf.score(X_f_val, y_f_val)
        cv_scores.append(fold_acc)
    
    print(f"\nTF-IDF + LR 5-Fold CV Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # Train final baseline on all training data
    final_clf = LogisticRegression(max_iter=1000, C=1.0)
    final_clf.fit(X_train_tfidf, y_train)
    
    y_pred_tfidf = final_clf.predict(X_test_tfidf)
    tfidf_acc = accuracy_score(y_test, y_pred_tfidf)
    tfidf_f1 = f1_score(y_test, y_pred_tfidf, average='macro')
    
    print("\n--- TF-IDF + Logistic Regression (Hold-out Test) ---")
    print(f"Accuracy: {tfidf_acc:.4f}")
    print(f"Macro F1: {tfidf_f1:.4f}")
    
    # Save models
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.joblib'))
    joblib.dump(final_clf, os.path.join(model_dir, 'tfidf_lr_model.joblib'))
    
    # Save results for comparison
    results = {
        'majority': {'acc': majority_acc, 'f1': majority_f1},
        'tfidf_lr': {'acc': tfidf_acc, 'f1': tfidf_f1, 'y_pred': y_pred_tfidf.tolist(), 'y_true': y_test.tolist()}
    }
    joblib.dump(results, os.path.join(model_dir, 'baseline_results.joblib'))
    
    print(f"\nBaseline models and results saved to {model_dir}")

if __name__ == "__main__":
    DATA_DIR = '/Users/ayushb3112/Documents/tdt4241/data_splits'
    MODEL_DIR = '/Users/ayushb3112/Documents/tdt4241/models'
    train_baselines(DATA_DIR, MODEL_DIR)
