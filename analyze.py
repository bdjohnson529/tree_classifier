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


def find_latest_config_file(results_dir, history_file):
    # Find the timestamp in the history file name
    import re
    match = re.search(r'training_history_(\d{8}_\d{6})', history_file)
    if not match:
        print("Could not extract timestamp from history file name.")
        return None
    timestamp = match.group(1)
    config_path = os.path.join(results_dir, f"config_snapshot_{timestamp}.py")
    if os.path.exists(config_path):
        return config_path
    else:
        print(f"No config snapshot found for timestamp {timestamp}.")
        return None


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


def get_ai_recommendation(history_df, config_path=None):
    # Read config.py for context
    if config_path is None:
        config_path = "src/config.py"
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
        "Please provide your recommendations as a concise list of bullet points, no more than 20 lines, and keep each point brief."
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


def analyze_all_runs(results_dir):
    """
    Collects all training_history_*.csv and matching config/model_comparison files,
    summarizes their key results, and sends to Claude for holistic recommendations.
    """
    import re
    history_files = sorted(glob.glob(os.path.join(results_dir, "training_history_*.csv")))
    runs = []
    for history_file in history_files:
        match = re.search(r'training_history_(\d{8}_\d{6})', history_file)
        if not match:
            continue
        timestamp = match.group(1)
        config_path = os.path.join(results_dir, f"config_snapshot_{timestamp}.py")
        model_comp_path = os.path.join(results_dir, f"model_comparison_{timestamp}.csv")
        try:
            history_df = pd.read_csv(history_file)
        except Exception as e:
            print(f"Could not read {history_file}: {e}")
            continue
        try:
            with open(config_path, "r") as f:
                config_code = f.read()
        except Exception:
            config_code = "(Missing config snapshot)"
        try:
            model_comp_df = pd.read_csv(model_comp_path) if os.path.exists(model_comp_path) else None
        except Exception:
            model_comp_df = None
        runs.append({
            "timestamp": timestamp,
            "history_file": history_file,
            "config_path": config_path,
            "config_code": config_code,
            "history_df": history_df,
            "model_comp_df": model_comp_df,
        })
    if not runs:
        print("No runs found in results directory.")
        return
    # Summarize for Claude
    summary = []
    for i, run in enumerate(runs):
        summary.append(f"=== Run {i+1} ({run['timestamp']}) ===")
        summary.append(f"Config (first 20 lines):\n" + '\n'.join(run['config_code'].splitlines()[:20]))
        summary.append("Training history (CSV head):")
        summary.append(run['history_df'].head(10).to_csv(index=False))
        if run['model_comp_df'] is not None:
            summary.append("Model comparison (CSV head):")
            summary.append(run['model_comp_df'].head(10).to_csv(index=False))
        summary.append("\n")
    summary_str = "\n".join(summary)
    if len(summary_str) > 12000:
        summary_str = summary_str[:12000] + "\n... (truncated)"
    prompt = (
        "You are an expert in deep learning model training. "
        "Below are the config.py snapshots and training histories from multiple experiments. "
        "Analyze the changes in config and their impact on training/validation accuracy and loss. "
        "Provide specific, actionable recommendations for tuning the hyperparameters in config.py "
        "to improve model performance in future runs. "
        "If you notice trends (e.g., certain changes improving or hurting results), highlight them.\n\n"
        f"{summary_str}\n"
        "Please provide your recommendations as a concise list of bullet points, no more than 20 lines, and keep each point brief."
    )
    api_url = os.environ.get("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        print("No Claude API key found in CLAUDE_API_KEY environment variable.")
        return
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    data = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 512,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            print("\nAI Recommendations (multi-run):\n")
            if "content" in result and isinstance(result["content"], list):
                print(result["content"][0].get("text", result["content"][0]))
            else:
                print(result)
        else:
            print("Error calling Claude API:", response.text)
    except Exception as e:
        print("Exception during Claude API call:", e)

    # Pause and allow user to enter a question
    user_question = input("\nYou may now enter a follow-up question for the AI (or press Enter to exit): ")
    if user_question.strip():
        followup_prompt = (
            "You are an expert in deep learning model training. "
            "Here is a follow-up question from the user about the previous experiments and recommendations. "
            f"Question: {user_question}\n"
            "Please answer concisely."
        )
        followup_data = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 512,
            "messages": [
                {"role": "user", "content": followup_prompt}
            ]
        }
        try:
            followup_response = requests.post(api_url, headers=headers, data=json.dumps(followup_data))
            if followup_response.status_code == 200:
                followup_result = followup_response.json()
                print("\nAI Follow-up Answer:\n")
                if "content" in followup_result and isinstance(followup_result["content"], list):
                    print(followup_result["content"][0].get("text", followup_result["content"][0]))
                else:
                    print(followup_result)
            else:
                print("Error calling Claude API for follow-up:", followup_response.text)
        except Exception as e:
            print("Exception during Claude API follow-up call:", e)
    else:
        print("Exiting.")


def main():
    results_dir = "results"
    history_file = find_latest_history_file(results_dir)

    '''
    if history_file:
        print(f"\nLoading: {history_file}")
        df = pd.read_csv(history_file)
        analyze_history(df)
        print("\n---\nCalling Claude for AI-powered recommendations...\n")
        config_path = find_latest_config_file(results_dir, history_file)
        get_ai_recommendation(df, config_path=config_path)
    '''

    # Uncomment the following line to analyze all runs in the results directory
    analyze_all_runs(results_dir)

if __name__ == "__main__":
    main()
