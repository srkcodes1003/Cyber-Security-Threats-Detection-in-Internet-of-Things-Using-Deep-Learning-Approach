import os
import argparse
import numpy as np
import pandas as pd
import time
from fetch_data import fetch_and_save_data
from preprocess import preprocess_pipeline, MULTICLASS_LABELS
from train import train_model
from evaluate import evaluate_performance, plot_learning_curves
from db_logger import init_db, log_threat, get_logs

def run_simulation(model, model_name: str, X_test, mode: str, dataset_name: str):
    """
    Simulates real-time threat detection on a batch of test packets.
    Feeds them individually into the NumPy model, computes inference time,
    and logs the predictions to the SQLite threat database.
    """
    print(f"\n[SIMULATION] Simulating real-time threat detection using {model_name} ({mode.upper()})...")
    
    # Select 10 random packets from the test set for simulation
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), size=min(10, len(X_test)), replace=False)
    
    for idx in sample_indices:
        packet = X_test[idx:idx+1]
        
        start_inf = time.time()
        pred_prob = model.predict(packet)
        inf_time = time.time() - start_inf
        
        # Decode prediction output
        if mode == "binary":
            prob = float(pred_prob[0][0])
            pred_class = "Malicious" if prob > 0.5 else "Normal"
            conf = prob if prob > 0.5 else (1.0 - prob)
        else:
            pred_idx = int(np.argmax(pred_prob[0]))
            pred_class = MULTICLASS_LABELS[pred_idx]
            conf = float(pred_prob[0][pred_idx])
            
        # Write to SQLite threat logs
        log_threat(
            source_dataset=dataset_name,
            predicted_class=pred_class,
            confidence_score=conf,
            execution_time=inf_time
        )
        time.sleep(0.05) # Brief pause to mimic live packets

def run_pipeline(mode: str, models_to_run, epochs: int, batch_size: int):
    """
    Executes the ingestion, preprocessing, training, evaluation, and DB logging pipeline.
    """
    csv_file = "kddcup99_sample.csv"
    
    # 1. Check if dataset exists, download if missing
    if not os.path.exists(csv_file):
        fetch_and_save_data()
        
    # 2. Preprocess data
    X_train, X_test, y_train, y_test, num_classes, feature_list = preprocess_pipeline(
        csv_file, mode=mode
    )
    
    class_names = ["Normal", "Malicious"] if mode == "binary" else MULTICLASS_LABELS
    results = {}
    
    # 3. Iterate and train selected models
    for model_type in models_to_run:
        # Train model
        model, history, train_time = train_model(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test, # Using stratified test set for validation during training
            y_val=y_test,
            num_classes=num_classes,
            mode=mode,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Generate learning curves
        plot_learning_curves(history, model_type.upper(), mode)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Evaluate performance & plot confusion matrix
        acc, prec, rec, f1, _ = evaluate_performance(
            y_test, y_pred, model_type.upper(), mode, class_names=class_names
        )
        
        results[model_type] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "training_time": train_time
        }
        
        # 4. Simulate real-time logging of prediction events
        run_simulation(model, model_type.upper(), X_test, mode, "KDDCup99_Sample")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="IoT Intrusion Detection Pipeline (NumPy DL Engine)")
    parser.add_argument("--mode", type=str, default="both", choices=["binary", "multiclass", "both"],
                        help="Classification mode: binary, multiclass, or both (default)")
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated models: ann, cnn, lstm, or all (default)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size (default: 128)")
    args = parser.parse_args()
    
    # Resolve models list
    if args.models.lower() == "all":
        models_to_run = ["ann", "cnn", "lstm"]
    else:
        models_to_run = [m.strip().lower() for m in args.models.split(",")]
        
    # Resolve modes
    if args.mode == "both":
        modes_to_run = ["binary", "multiclass"]
    else:
        modes_to_run = [args.mode]
        
    # Initialize SQLite database
    init_db()
    
    overall_summary = []
    
    for mode in modes_to_run:
        print("\n" + "#"*60)
        print(f" STARTING PIPELINE IN {mode.upper()} MODE ")
        print("#"*60)
        
        mode_results = run_pipeline(mode, models_to_run, args.epochs, args.batch_size)
        
        for m_type, metrics in mode_results.items():
            metrics["mode"] = mode
            metrics["model"] = m_type.upper()
            overall_summary.append(metrics)
            
    # Print final project comparisons table
    summary_df = pd.DataFrame(overall_summary)
    print("\n" + "="*80)
    print(" PROJECT RESULTS COMPARISON BENCHMARK ")
    print("="*80)
    print(summary_df.to_string(index=False, formatters={
        "accuracy": "{:.2%}".format,
        "precision": "{:.2%}".format,
        "recall": "{:.2%}".format,
        "f1_score": "{:.2%}".format,
        "training_time": "{:.2f}s".format
    }))
    print("="*80)
    
    # Read and print sample database records
    print("\n[DB] Querying recent entries from threat_logs database:")
    db_records = get_logs()
    print(f"[DB] Total threat logs recorded: {len(db_records)}")
    print(f"{'Log ID':<8} | {'Timestamp':<19} | {'Dataset':<15} | {'Class':<12} | {'Confidence':<10} | {'Latency':<8}")
    print("-" * 85)
    for record in db_records[:10]: # Print top 10 logs
        log_id, ts, dataset, cls, conf, lat = record
        print(f"{log_id:<8} | {ts:<19} | {dataset:<15} | {cls:<12} | {conf:<10.4f} | {lat:<8.4f}s")
    print("="*85)

if __name__ == "__main__":
    main()
