import pandas as pd
import os
import glob
from sklearn.metrics import f1_score

# CONFIG
SUBMISSION_DIR = "submissions"
TRUTH_FILE = "data/test_labels_hidden.csv"
LEADERBOARD_CSV = "leaderboard/leaderboard.csv"
# FIXED: Points to the root file so the README link works
LEADERBOARD_MD = "LEADERBOARD.md"

def main():
    # 1. Load Truth
    if not os.path.exists(TRUTH_FILE):
        print(f"❌ Error: Truth file {TRUTH_FILE} not found")
        return
    
    true_df = pd.read_csv(TRUTH_FILE).sort_values('id').reset_index(drop=True)
    
    # 2. Find All Submissions
    files = glob.glob(f"{SUBMISSION_DIR}/**/*.csv", recursive=True)
    results = []

    print(f"🔍 Found {len(files)} submission files.")

    for file_path in files:
        if "sample_submission.csv" in file_path:
            continue
            
        try:
            pred_df = pd.read_csv(file_path)
            
            # Validation: check for required columns
            if 'id' not in pred_df.columns or 'y_pred' not in pred_df.columns:
                print(f"⚠️ Skipping {file_path}: Missing columns 'id' or 'y_pred'. Found: {list(pred_df.columns)}")
                continue 
            
            # Validation: IDs match
            pred_df = pred_df.sort_values('id').reset_index(drop=True)
            if not pred_df['id'].equals(true_df['id']):
                print(f"⚠️ Skipping {file_path}: ID mismatch with ground truth.")
                continue

            # Score (Macro F1)
            score = f1_score(true_df['label'], pred_df['y_pred'], average='macro')
            
            filename = os.path.basename(file_path)
            team_name = os.path.splitext(filename)[0]

            if team_name.lower() in ['predictions', 'submission', 'my_submission']:
                team_name = os.path.basename(os.path.dirname(file_path))

            print(f"✅ Scored {team_name}: {score:.4f}") # Now we can see the score in logs!

            results.append({
                'team': team_name,
                'score': score,
                'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
            })

        except Exception as e:
            print(f"⚠️ Failed to process {file_path}: {e}")

    # 3. Save Leaderboard CSV
    if not results:
        print("No valid submissions found.")
        return

    df = pd.DataFrame(results)
    
    # Logic: Keep ONLY the best score per team
    original_count = len(df)
    df = df.sort_values(by='score', ascending=False)
    df = df.drop_duplicates(subset=['team'], keep='first')
    
    if len(df) < original_count:
        print(f"ℹ️ Note: Removed {original_count - len(df)} duplicate/lower scores for the same team name.")

    os.makedirs("leaderboard", exist_ok=True)
    df.to_csv(LEADERBOARD_CSV, index=False)
    
    render_markdown(df)
    print("🚀 Leaderboard Update Complete!")

def render_markdown(df):
    md = "# 🏆 Tumor Diagnosis Leaderboard\n\n"
    md += "| Rank | Team | Macro F1 Score | Last Updated |\n"
    md += "| :--- | :--- | :--- | :--- |\n"
    
    # Add Rank with Dense logic (1, 2, 2, 3)
    df['rank'] = df['score'].rank(method='dense', ascending=False).astype(int)
    
    for _, row in df.iterrows():
        r = row['rank']
        medal = "🥇" if r == 1 else "🥈" if r == 2 else "🥉" if r == 3 else str(r)
        # FIXED: Format score to 4 decimal places
        md += f"| {medal} | {row['team']} | {row['score']:.4f} | {row['date']} |\n"
        
    # Write to ROOT directory
    with open(LEADERBOARD_MD, "w") as f:
        f.write(md)

if __name__ == "__main__":
    main()
