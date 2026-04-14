import pandas as pd
import numpy as np
import datetime
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Ground truth timestamps for Friday-WorkingHours from user input
# All times are in EDT (UTC-4), PCAP is typically captured in ADT/AST or UTC-4
# PCAP start timestamp based on typical Friday PCAP: 2017-07-07 08:00:00 (approx)
# Let's map these to start and end times

ATTACKS = [
    # Botnet ARES (10:02 a.m. - 11:02 a.m.)
    ("2017-07-07 10:02:00", "2017-07-07 11:02:00", "Botnet ARES"),
    
    # Port Scan: Firewall Rule on
    ("2017-07-07 13:55:00", "2017-07-07 13:57:00", "Port Scan"),
    ("2017-07-07 13:58:00", "2017-07-07 14:00:00", "Port Scan"),
    ("2017-07-07 14:01:00", "2017-07-07 14:04:00", "Port Scan"),
    ("2017-07-07 14:05:00", "2017-07-07 14:07:00", "Port Scan"),
    ("2017-07-07 14:08:00", "2017-07-07 14:10:00", "Port Scan"),
    ("2017-07-07 14:11:00", "2017-07-07 14:13:00", "Port Scan"),
    ("2017-07-07 14:14:00", "2017-07-07 14:16:00", "Port Scan"),
    ("2017-07-07 14:17:00", "2017-07-07 14:19:00", "Port Scan"),
    ("2017-07-07 14:20:00", "2017-07-07 14:21:00", "Port Scan"),
    ("2017-07-07 14:22:00", "2017-07-07 14:24:00", "Port Scan"),
    ("2017-07-07 14:33:00", "2017-07-07 14:33:59", "Port Scan"),
    ("2017-07-07 14:35:00", "2017-07-07 14:35:59", "Port Scan"),
    
    # Port Scan: Firewall rules off
    ("2017-07-07 14:51:00", "2017-07-07 14:53:00", "Port Scan"),
    ("2017-07-07 14:54:00", "2017-07-07 14:56:00", "Port Scan"),
    ("2017-07-07 14:57:00", "2017-07-07 14:59:00", "Port Scan"),
    ("2017-07-07 15:00:00", "2017-07-07 15:02:00", "Port Scan"),
    ("2017-07-07 15:03:00", "2017-07-07 15:05:00", "Port Scan"),
    ("2017-07-07 15:06:00", "2017-07-07 15:07:00", "Port Scan"),
    ("2017-07-07 15:08:00", "2017-07-07 15:10:00", "Port Scan"),
    ("2017-07-07 15:11:00", "2017-07-07 15:12:00", "Port Scan"),
    ("2017-07-07 15:13:00", "2017-07-07 15:15:00", "Port Scan"),
    ("2017-07-07 15:16:00", "2017-07-07 15:18:00", "Port Scan"),
    ("2017-07-07 15:19:00", "2017-07-07 15:21:00", "Port Scan"),
    ("2017-07-07 15:22:00", "2017-07-07 15:24:00", "Port Scan"),
    ("2017-07-07 15:25:00", "2017-07-07 15:25:59", "Port Scan"),
    ("2017-07-07 15:26:00", "2017-07-07 15:27:00", "Port Scan"),
    ("2017-07-07 15:28:00", "2017-07-07 15:29:00", "Port Scan"),
    
    # DDoS LOIT
    ("2017-07-07 15:56:00", "2017-07-07 16:16:00", "DDoS")
]

def main():
    import pickle
    
    base_dir = r"C:\Users\Acer\Downloads\quantum_sniffer (2) (1)\quantum_sniffer\quantum_sniffer"
    cache_path = os.path.join(base_dir, "pcap_training_results", "features_cache.pkl")
    json_path = os.path.join(base_dir, "pcap_training_results", "model_baseline.json")
    csv_path = os.path.join(base_dir, "pcap_training_results", "window_scores.csv")
    
    with open(json_path, 'r') as f:
        baseline = json.load(f)
        
    threshold = baseline['score_stats']['threshold']
    scores_df = pd.read_csv(csv_path)
    
    with open(cache_path, 'rb') as f:
        cached_data = pickle.load(f)
        
    features, metadata_list = cached_data
    
    # The timestamps in PCAP (e.g. 1499428779) correspond to 11:59:39 UTC.
    # The attacks are documented in UTC-4 (e.g., morning and afternoon of July 7th 2017).
    # We will convert attack times to UTC timestamps to compare against the PCAP timestamps.
    attack_intervals = []
    
    # Define EDT timezone (UTC-4)
    edt_tz = datetime.timezone(datetime.timedelta(hours=-4))
    
    for start_str, end_str, label in ATTACKS:
        start_t = datetime.datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=edt_tz).timestamp()
        end_t = datetime.datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=edt_tz).timestamp()
        attack_intervals.append((start_t, end_t, label))
    
    y_true = []
    y_pred = []
    
    for i in range(len(scores_df)):
        score = scores_df.iloc[i]['score']
        pred = 1 if score > threshold else 0
        y_pred.append(pred)
        
        meta = metadata_list[i]
        window_start = meta['start_ts']
        window_end = meta['end_ts']
        
        is_attack = 0
        for atk_start, atk_end, atk_label in attack_intervals:
            if window_start <= atk_end and atk_start <= window_end:
                is_attack = 1
                break
                
        y_true.append(is_attack)
        
    # Calculate metrics
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    specificity = tn / (tn + fp)
    fpr = fp / (tn + fp)
    fnr = fn / (tp + fn)
    balanced_acc = (recall + specificity) / 2
    
    print("\n--- PERFORMANCE METRICS ---")
    print(f"Total Samples: {len(y_true)}")
    print(f"Actual Positive (Attacks): {sum(y_true)}")
    print(f"Actual Negative (Normal): {len(y_true) - sum(y_true)}")
    print(f"Prevalence: {sum(y_true)/len(y_true)*100:.1f}%\n")
    
    print(f"True Positives  (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives  (TN): {tn}")
    print(f"False Negatives (FN): {fn}\n")
    
    print(f"Recall:              {recall*100:.1f}%")
    print(f"Precision:           {precision*100:.1f}%")
    print(f"F1 Score:            {f1*100:.1f}%")
    print(f"Accuracy:            {accuracy*100:.1f}%")
    print(f"Specificity:         {specificity*100:.1f}%")
    print(f"False Positive Rate: {fpr*100:.1f}%")
    print(f"False Negative Rate: {fnr*100:.1f}%")
    print(f"MCC:                 {mcc:.4f}")
    print(f"Balanced Accuracy:   {balanced_acc*100:.1f}%")
    
    # Let's save a visual representation exactly like the user requested
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], annot_kws={"size": 16})
    axes[0].set_title('Confusion Matrix', fontsize=16)
    axes[0].set_xlabel('Predicted Label', fontsize=14)
    axes[0].set_ylabel('True Label', fontsize=14)
    axes[0].xaxis.set_ticklabels(['Normal (0)', 'Attack (1)'])
    axes[0].yaxis.set_ticklabels(['Normal (0)', 'Attack (1)'])
    
    # 2. Key Metrics Bar Chart
    metrics_names = ['Recall', 'Precision', 'F1 Score', 'Accuracy', 'Specificity']
    metrics_vals = [recall, precision, f1, accuracy, specificity]
    
    bars = axes[1].bar(metrics_names, metrics_vals, color=['#e74c3c', '#f39c12', '#f39c12', '#2ecc71', '#2ecc71'])
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title('Key Performance Metrics', fontsize=16)
    axes[1].set_ylabel('Score', fontsize=14)
    
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval*100:.1f}%', ha='center', va='bottom', fontsize=12)
        
    plt.tight_layout()
    output_png = os.path.join(base_dir, "pcap_training_results", "confusion_matrix.png")
    plt.savefig(output_png)
    print(f"\nSaved visualization to {output_png}")

if __name__ == '__main__':
    main()
