import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import os

def prepare_data(input_file, output_dir):
    # Load dataset
    df = pd.read_csv(input_file)
    
    # Combine title and full post for feature extraction
    # Fill NaN values if any
    df['title'] = df['title'].fillna('')
    df['full post'] = df['full post'].fillna('')
    df['text'] = df['title'] + " " + df['full post']
    
    # Map labels: user_is_fault -> 1, user_ok -> 0
    label_map = {'user_is_fault': 1, 'user_ok': 0}
    df['label'] = df['verdict'].map(label_map)
    
    # Select necessary columns
    df = df[['text', 'label', 'pid']]
    
    # 80/20 train/test split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Re-index train_df to ensure clean fold assignment
    train_df = train_df.reset_index(drop=True)
    
    # Stratified 5-Fold Cross-Validation on the training portion
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_df['fold'] = -1
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df['text'], train_df['label'])):
        train_df.loc[val_idx, 'fold'] = fold
        
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save split datasets
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Data prepared and saved to {output_dir}")
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Fold distribution in train set:\n{train_df['fold'].value_counts()}")

if __name__ == "__main__":
    INPUT_FILE = '/Users/ayushb3112/Documents/tdt4241/aita_verdicts_unique_6000.csv'
    OUTPUT_DIR = '/Users/ayushb3112/Documents/tdt4241/data_splits'
    prepare_data(INPUT_FILE, OUTPUT_DIR)
