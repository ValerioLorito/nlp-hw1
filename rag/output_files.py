import os
import json

def create_jsonl_file(results, filepath, type):
    with open(filepath, 'w', encoding='utf-8') as f:
        for result in results:
            if type == "generated_responses":
                json_record = {
                    "query_id": result["query_id"],
                    "retrieved_chunks": result["retrieved_chunks"],
                    "augmented_prompt": result["augmented_prompt"],
                    "generated_answer": result["generated_answer"]
                }
            elif type == "LLM_judge":
                json_record = {
                    "query_id": result["query_id"],
                    "retrieved_chunks": result["retrieved_chunks"],
                    "augmented_prompt": result["augmented_prompt"],
                    "generated_answer": result["generated_answer"],
                    "llm_judge": result["llm_judge_output"],
                    "annotator_1": result["annotator_1"],
                    "annotator_2": result["annotator_2"]
                }

            f.write(json.dumps(json_record) + '\n')

def generate_jsonl_file(results, split_name, model_name, setting, type):
    group_name = "Its_always_loss"

    output_dir = os.path.join("HW2", split_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = f"{group_name}-{split_name}-{model_name}-{setting}.jsonl"
    filepath = os.path.join(output_dir, filename)
    print(f"{filename} generation...")
    create_jsonl_file(results, filepath, type)
    

