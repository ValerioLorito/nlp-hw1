import json

def compare_jsonl(f1, f2):
    with open(f1) as file1, open(f2) as file2:
        for line1, line2 in zip(file1, file2):
            d1 = json.loads(line1)
            d2 = json.loads(line2)
            for key in d1:
                if d1[key] != d2.get(key):
                    print(f"Difference found for key : {key}")

def main():
    f1 = "f1"
    f2 = "f2"
    compare_jsonl(f1, f2)



if __name__ == "__main__":
  main()