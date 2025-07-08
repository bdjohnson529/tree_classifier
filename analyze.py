import requests
import json
import os
import glob
import pandas as pd
from dotenv import load_dotenv

load_dotenv("secrets.env")  # Load environment variables from secrets.env


def find_latest_history_file(results_dir):
    files = glob.glob(os.path.join(results_dir, "training_history_*.csv"))
    if not files:
        print("No training history CSV files found.")
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def analyze_history(df):
    # Assume only one model for simplicity
    model_name = df['model'].iloc[0]
    train_acc = df['train_accuracy']
    val_acc = df['val_accuracy']
    train_loss = df['train_loss']
    val_loss = df['val_loss']

    print(f"\nAnalyzing training history for model: {model_name}")
    print(f"Final training accuracy: {train_acc.iloc[-1]:.4f}")
    print(f"Final validation accuracy: {val_acc.iloc[-1]:.4f}")
    print(f"Final training loss: {train_loss.iloc[-1]:.4f}")
    print(f"Final validation loss: {val_loss.iloc[-1]:.4f}")

    # Recommendations
    print("\nRecommendations:")
    if train_acc.iloc[-1] > 0.90 and val_acc.iloc[-1] < 0.7:
        print("- Overfitting detected: Increase DROPOUT_RATE, add more data augmentation, or reduce model complexity.")
    elif train_acc.iloc[-1] < 0.7 and val_acc.iloc[-1] < 0.7:
        print("- Underfitting detected: Try decreasing DROPOUT_RATE, increasing model complexity, or increasing INITIAL_EPOCHS.")
    elif val_acc.max() - val_acc.iloc[-1] > 0.05:
        print("- Validation accuracy is decreasing: Try reducing INITIAL_LEARNING_RATE or using early stopping.")
    elif val_acc.iloc[-1] > 0.8:
        print("- Model is performing well! Consider fine-tuning more layers or increasing FINE_TUNE_EPOCHS for further improvement.")
    else:
        print("- If accuracy is not improving, try adjusting INITIAL_LEARNING_RATE, FINE_TUNE_LEARNING_RATE, or unfreezing more layers.")


def get_ai_recommendation(history_df, config_path="src/config.py"):
    # Read config.py for context
    try:
        with open(config_path, "r") as f:
            config_code = f.read()
    except Exception as e:
        config_code = "(Could not read config.py: " + str(e) + ")"

    # Convert history to CSV string (truncate if too large)
    history_csv = history_df.to_csv(index=False)
    if len(history_csv) > 8000:
        history_csv = history_csv[:8000] + "\n... (truncated)"

    prompt = (
        "You are an expert in deep learning model training. "
        "Given the following training history (CSV) and config.py, "
        "analyze the results and provide specific, actionable recommendations "
        "for tuning the hyperparameters in config.py to improve model performance.\n\n"
        "config.py:\n"
        f"{config_code}\n\n"
        "Training history (CSV):\n"
        f"{history_csv}\n\n"
        "Please provide your recommendations in a clear, concise list."
    )

    # Call Claude or other LLM API (replace with your actual endpoint and API key)
    api_url = os.environ.get("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")
    api_key = os.environ.get("CLAUDE_API_KEY")  # Set your API key in env variable

    if not api_key:
        print("No Claude API key found in CLAUDE_API_KEY environment variable.")
        return

    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    data = {
        "model": "claude-3-opus-20240229",  # Or your available Claude model
        "max_tokens": 512,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            print("\nAI Recommendations:\n")
            # Claude's response format may vary; adjust as needed
            if "content" in result and isinstance(result["content"], list):
                print(result["content"][0].get("text", result["content"][0]))
            else:
                print(result)
        else:
            print("Error calling Claude API:", response.text)
    except Exception as e:
        print("Exception during Claude API call:", e)


def main():
    results_dir = "results"
    history_file = find_latest_history_file(results_dir)
    if history_file:
        print(f"\nLoading: {history_file}")
        df = pd.read_csv(history_file)
        analyze_history(df)
        print("\n---\nCalling Claude for AI-powered recommendations...\n")
        get_ai_recommendation(df)

if __name__ == "__main__":
    main()
