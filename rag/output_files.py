import os
import json

def create_jsonl_file(results, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for result in results:
            json_record = {
                "query_id": result["query_id"],
                "retrieved_chunks": result["retrieved_chunks"],
                "augmented_prompt": result["augmented_prompt"],
                "generated_answer": result["generated_answer"]
            }
            f.write(json.dumps(json_record) + '\n')

def generate_jsonl_file(results, split_name, model_name, setting):
    group_name = "Its_always_loss"

    output_dir = os.path.join("HW2", split_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = f"{group_name}-{split_name}-{model_name}-{setting}.jsonl"
    filepath = os.path.join(output_dir, filename)
    print(f"{filename} generation...")
    create_jsonl_file(results, filepath)
    


