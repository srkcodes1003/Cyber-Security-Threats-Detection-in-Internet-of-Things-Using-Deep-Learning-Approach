import time
import os
import joblib
from models import build_lstm

# Ensure models directory exists
MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_model(model_type: str, X_train, y_train, X_val, y_val, num_classes: int, mode: str = "binary", epochs: int = 10, batch_size: int = 64):
    """
    Orchestrates training using the NumPy deep learning engine.
    1. Instantiates the custom model.
    2. Runs training loop via NumPy fit interface.
    3. Saves model object to disk using joblib.
    """
    print(f"\n[TRAINER] Initializing training for: {model_type.upper()} ({mode.upper()} mode) using NumPy Engine")
    
    input_dim = X_train.shape[1]
    
    if model_type.lower() == "lstm":
        model = build_lstm(input_dim, num_classes)
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Only LSTM is supported.")
        
    start_time = time.time()
    
    # Run fit loop
    history = model.fit(
        X_train=X_train,
        y_train=y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    execution_time = time.time() - start_time
    print(f"[TRAINER] Completed training {model_type.upper()} in {execution_time:.2f} seconds.")
    
    # Save the custom model object using joblib
    save_path = os.path.join(MODELS_DIR, f"{model_type.lower()}_{mode}_model.joblib")
    joblib.dump(model, save_path)
    print(f"[TRAINER] Model successfully saved to: {save_path}")
    
    return model, history, execution_time
