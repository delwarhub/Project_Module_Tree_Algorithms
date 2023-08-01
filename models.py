import os
import pickle
from utils import config
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# train a RandomForestClassifer 
def train_RandomForestClassifier(X_train, y_train, 
                                 n_estimator: int=100,
                                 model_name: str="100_RFC"):
    
    # Train RandomForestClassifier
    print(f"Training a RandomForestClassifir with n_estimator: {n_estimator}")
    model = RandomForestClassifier(n_estimators=n_estimator, 
                                   random_state=42)
    model.fit(X_train, y_train)
    
    # save model for future use
    path_to_save_loc = os.path.join(config["PATH_TO_MODEL_OUTPUT_DIR"], f"{model_name}.pkl")
    pickle.dump(model, open(path_to_save_loc, 'wb'))
    model_ = pickle.load(open(path_to_save_loc, 'rb'))
    return model_

# load already trained model
def load_trained_model(path_to_save_loc: str):
    model_ = pickle.load(open(path_to_save_loc, 'rb'))
    return model_

# perform evaluation on already trained model; i.e. accuracy, & classification_report
def evaluate(model, X_test,
             y_test, labels):
    
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=labels))