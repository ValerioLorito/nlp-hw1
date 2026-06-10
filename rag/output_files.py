import os
import json
import pandas as pd

def create_jsonl_file(results, filepath, type):
    with open(filepath, 'w', encoding='utf-8') as f:
        for result in results:
            if type == "generated_responses":
                json_record = {
                    "query_id": result["query_id"],
                    "augmented_prompt": result["augmented_prompt"] if result.get("augmented_prompt") is not None else "",
                    "retrieved_chunks": result["retrieved_chunks"] if result.get("retrieved_chunks") is not None else [],
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

    print(f"JSONL file created at: {filepath}")

def generate_jsonl_file(results, split_name, model_name, setting, type):
    group_name = "Its_always_loss"

    output_dir = split_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    filename = f"{group_name}-{split_name}-{model_name}-{setting}.jsonl"
    print(f"{filename} generation...")
    filepath = os.path.join(output_dir, filename)
    create_jsonl_file(results, filepath, type)

def export_judge_to_excel(jsonl_filepath, excel_filepath, queries, short_answers):
    records = []
    
    with open(jsonl_filepath, 'r', encoding='utf-8') as f:
        for line, query, short_answer in zip(f, queries, short_answers):
            if line.strip():
                record = json.loads(line)
                row = {
                    "query_id": record.get("query_id", ""),
                    "query": query,
                    "retrieved_chunks": str(record.get("retrieved_chunks", [])),
                    "augmented_prompt": record.get("augmented_prompt", ""),
                    "generated_answer": record.get("generated_answer", ""),
                    "short_answer": short_answer[0],
                    "llm_judge": record.get("llm_judge", 0),
                    "annotator_1": "", 
                    "annotator_2": "" 
                }
                records.append(row)
                
    df = pd.DataFrame(records)
    
    df.to_excel(excel_filepath, sheet_name="Jugements", index=False)
    
    print(f"excel file saved : {excel_filepath}")

def export_annotations(excel_filepath, jsonl_filepath):
    df = pd.read_excel(excel_filepath, sheet_name="Jugements")
    
    records = []
    for _, row in df.iterrows():
        record = {
            "query_id": row.get("query_id", ""),
            "retrieved_chunks": row.get("retrieved_chunks", ""),
            "augmented_prompt": row.get("augmented_prompt", ""),
            "generated_answer": row.get("generated_answer", ""),
            "llm_judge": row.get("llm_judge", 0),
            "annotator_1": row.get("annotator_1", ""),
            "annotator_2": row.get("annotator_2", "")
        }
        records.append(record)
    
    with open(jsonl_filepath, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"jsonl file saved : {jsonl_filepath}")