import json
from src.data_loader import load_data # Ensure the path is correct

def load_jsonl(filepath):
    """Reads a JSONL file and returns a list of dictionaries."""
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

def main():
    print("Loading dataset, queries, and candidates...")
    ds = load_data()
    test_queries = ds["test"]["query"]
    answer_positions = ds["test"]["answer_pos"]
    
    # Load candidate chunks to extract the answer text
    candidate_chunks = ds["test"]["candidate_chunks"] 

    # Insert the correct paths to your JSONL files
    mini_file = "Its_always_loss-test-all-miniLM-L6-v2-2-mnr-cosine.jsonl"
    f2llm_file = "Its_always_loss-test-F2LLM-v2-80M-1-mnr-cosine.jsonl"

    mini_data = load_jsonl(mini_file)
    f2llm_data = load_jsonl(f2llm_file)

    mini_wins = []
    f2llm_wins = []

    # Iterate simultaneously over everything: JSONL 1, JSONL 2, correct positions, query text, and candidate lists
    for i, (q_mini, q_f2llm, ans_pos, query_text, chunks) in enumerate(zip(mini_data, f2llm_data, answer_positions, test_queries, candidate_chunks)):

        # Extract the Query ID (it's the single key of the dictionary)
        query_id = list(q_mini.keys())[0]
        
        # Extract the correct answer text using ans_pos
        correct_answer_text = chunks[ans_pos]

        # Extract the predicted ranking lists
        ranking_mini = list(q_mini.values())[0]
        ranking_f2llm = list(q_f2llm.values())[0]

        # Find the position (rank) of the correct answer in the predicted lists
        # We add +1 because Python is 0-indexed, but real-world ranks start from 1
        try:
            rank_mini = ranking_mini.index(ans_pos) + 1
        except ValueError:
            rank_mini = 999  # Not found

        try:
            rank_f2llm = ranking_f2llm.index(ans_pos) + 1
        except ValueError:
            rank_f2llm = 999  # Not found

        # CONDITION: We want cases where one model put it in the Top 10 and the other didn't
        if rank_mini <= 10 and rank_f2llm > 10:
            mini_wins.append((i, query_id, query_text, correct_answer_text, rank_mini, rank_f2llm))
        elif rank_f2llm <= 10 and rank_mini > 10:
            f2llm_wins.append((i, query_id, query_text, correct_answer_text, rank_mini, rank_f2llm))

    # Sort the lists to get the "worst failures" of the defeated model
    # x[5] is rank_f2llm, x[4] is rank_mini. reverse=True sorts from highest (worst) to lowest rank
    worst_f2llm_failures = sorted(mini_wins, key=lambda x: x[5], reverse=True)[:5]
    worst_mini_failures = sorted(f2llm_wins, key=lambda x: x[4], reverse=True)[:5]

    # ---- PRINT RESULTS FOR ERROR ANALYSIS ----
    print("\n" + "="*80)
    print(" DIFFERENCE ANALYSIS (TOP 5 WORST GAPS)")
    print("="*80)
    
    print(f"\n🏆 all-MiniLM in Top 10 | Top 5 worst F2LLM failures (out of {len(mini_wins)} total cases)")
    print("-" * 80)
    for idx, q_id, text, ans_text, r_mini, r_f2llm in worst_f2llm_failures:
        print(f"🔍 Index: {idx} | Query ID: {q_id}")
        print(f"📈 Rank -> all-MiniLM: {r_mini} | F2LLM: {r_f2llm}")
        print(f"🗣️  Query:   '{text}'")
        print(f"✅ Answer:  '{ans_text}'\n")

    print(f"\n🏆 F2LLM in Top 10 | Top 5 worst all-MiniLM failures (out of {len(f2llm_wins)} total cases)")
    print("-" * 80)
    for idx, q_id, text, ans_text, r_mini, r_f2llm in worst_mini_failures:
        print(f"🔍 Index: {idx} | Query ID: {q_id}")
        print(f"📈 Rank -> F2LLM: {r_f2llm} | all-MiniLM: {r_mini}")
        print(f"🗣️  Query:   '{text}'")
        print(f"✅ Answer:  '{ans_text}'\n")

if __name__ == "__main__":
    main()