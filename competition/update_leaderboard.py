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
    # Recursive search for any csv file in submissions/
    files = glob.glob(f"{SUBMISSION_DIR}/**/*.csv", recursive=True)
    results = []

    print(f"🔍 Found {len(files)} submission files.")

    for file_path in files:
        # Skip the example file
        if "sample_submission.csv" in file_path:
            continue
            
        try:
            # Load Submission
            pred_df = pd.read_csv(file_path)
            
            # FIXED: Changed 'y_pred' to 'label' to match your README instructions
            if 'id' not in pred_df.columns or 'y_pred' not in pred_df.columns:
                print(f"⚠️ Skipping {file_path}: Missing columns (id, y_pred)")
                continue 
            
            # Validation: IDs match
            pred_df = pred_df.sort_values('id').reset_index(drop=True)
            if not pred_df['id'].equals(true_df['id']):
                print(f"⚠️ Skipping {file_path}: ID mismatch")
                continue

            # Score (Macro F1)
            # FIXED: Using 'label' column
            score = f1_score(true_df['y_pred'], pred_df['y_pred'], average='macro')
            
            # Extract Team Name
            # Logic: If file is 'submissions/TeamA.csv', team is 'TeamA'
            filename = os.path.basename(file_path)
            team_name = os.path.splitext(filename)[0]

            # Cleanup: If name is generic like 'predictions', try folder name
            if team_name.lower() in ['predictions', 'submission', 'my_submission']:
                team_name = os.path.basename(os.path.dirname(file_path))

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

    # Create dataframe
    df = pd.DataFrame(results)
    
    # Sort by Score (Descending) and keep best score per team
    df = df.sort_values(by='score', ascending=False)
    # If a team submitted multiple times, keep their BEST score
    df = df.drop_duplicates(subset=['team'], keep='first')
    
    # Ensure directory exists for the CSV backup
    os.makedirs("leaderboard", exist_ok=True)
    df.to_csv(LEADERBOARD_CSV, index=False)
    
    # 4. Render Markdown
    render_markdown(df)
    print("✅ Leaderboard Updated!")

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
