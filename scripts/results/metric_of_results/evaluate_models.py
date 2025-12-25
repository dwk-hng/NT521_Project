import os
import glob
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def evaluate_models():
    # The script is now located in the same directory as the results
    results_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(results_dir, 'model_evaluation_summary.csv')
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
    
    summary_data = []
    
    print(f"Found {len(csv_files)} result files in {results_dir}\n")
    
    for file_path in sorted(csv_files):
        filename = os.path.basename(file_path)
        
        # Skip the output file if it exists in the list
        if filename == 'model_evaluation_summary.csv':
            continue

        # Extract model name (everything before the date part '_2025')
        if '_2025' in filename:
            model_name = filename.split('_2025')[0]
        else:
            model_name = filename.replace('.csv', '')
            
        try:
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            if 'Ground_true' not in df.columns or 'Prediction' not in df.columns:
                print(f"Skipping {filename}: Missing 'Ground_true' or 'Prediction' columns.")
                continue
                
            y_true = df['Ground_true']
            y_pred = df['Prediction']
            
            # Calculate metrics
            # Assuming 'Malicious' is the positive class. 
            # We need to check the unique values to be sure or specify pos_label.
            # Based on the file content provided in context, values are 'Benign' and 'Malicious'.
            pos_label = 'Malicious'
            
            precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            
            # Print results to terminal
            print(f"--- Evaluation for Model: {model_name} ---")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print("-" * 40 + "\n")
            
            summary_data.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save summary to file
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_file, index=False)
        print(f"Summary of evaluation saved to: {output_file}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    evaluate_models()
