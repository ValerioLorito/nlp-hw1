import os
import json
import pandas as pd

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

            f.write(json.dumps(json_record, ensure_ascii=False) + '\n')

def generate_jsonl_file(results, split_name, model_name, setting, type):
    group_name = "Its_always_loss"

    output_dir = os.path.join("HW2", split_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = f"{group_name}-{split_name}-{model_name}-{setting}.jsonl"
    filepath = os.path.join(output_dir, filename)
    print(f"{filename} generation...")
    create_jsonl_file(results, filepath, type)

def export_judge_to_excel(jsonl_filepath, excel_filepath):
    records = []
    
    with open(jsonl_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                row = {
                    "Query ID": record.get("query_id", ""),
                    "Chunks": str(record.get("retrieved_chunks", [])),
                    "Augmented Prompt": record.get("augmented_prompt", ""),
                    "Generated answer": record.get("generated_answer", ""),
                    "LLM Judge Score": record.get("llm_judge", 0),
                    "Annotator 1": "", 
                    "Annotator 2": "" 
                }
                records.append(row)
                
    df = pd.DataFrame(records)
    
    df.to_excel(excel_filepath, sheet_name="Jugements", index=False)
    
    print(f"excel file saved : {excel_filepath}")
