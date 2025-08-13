import os
import json
import random
import time
from openai import OpenAI

# --- CONFIGURATION ---

# 1. API
api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(
#     base_url="http://localhost:11434/v1", api_key="ollama"  # Standard Ollama endpoint
# )
client = OpenAI(
    api_key= api_key
)

# 2. Load list of religions
MASTER_RELIGION_LIST = []
with open("all_rels_da.txt", "r", encoding="utf-8") as f:
    MASTER_RELIGION_LIST = f.readlines()
MASTER_RELIGION_LIST = [line.strip() for line in MASTER_RELIGION_LIST]

# 3. Output
OUTPUT_FILENAME = "finetuning_dataset_gpt4.jsonl"


def generate_system_prompt():
    """Defines the core instruction for the language model."""
    # This is the detailed instruction set from your original prompt.
    return """
You are a religious studies expert. Give me a short descriptive paragraph of the following religions. Provide information on the dimensions Metaphysics, Cosmology, Anthropology, Soteriology, Ethics & Morality, Praxis, and Epistemology for each religion. Provide a score for each dimension for each religion and calculate the distance between each religion based on the scores.

The scoring rubric for each dimension is on a 1-5 scale:
* Metaphysics: 1 (Purely Materialist) → 5 (Strongly Transcendent/Supernatural)
* Cosmology: 1 (Purely Naturalistic) → 5 (Direct Supernatural Creation)
* Anthropology: 1 (Purely Biological) → 5 (Possessing a Divine/Eternal Soul)
* Soteriology: 1 (This-Worldly/Self-Achieved) → 5 (Requires Divine Grace/Intervention)
* Ethics: 1 (Human-Derived/Situational) → 5 (Divinely Commanded/Absolute)
* Praxis: 1 (Minimal/No Ritual) → 5 (Highly Prescribed Rituals)
* Epistemology: 1 (Purely Empirical/Rational) → 5 (Primarily based on Divine Revelation)
* Ecclesiology: 1 (Decentralized / Individualistic) → 5 (Rigid Hierarchy)
* Eschatology: 1 (Cyclical / A-historical) → 5 (Aggressively Universalist)
* Exclusivity: 1 (Pluralist) → 3 (Inclusivist) → 5 (Exclusivist)


Generate a response that includes a descriptive paragraph for each religion, a markdown table with scores for each dimension, and a final markdown table with the calculated Euclidean distance matrix between the religions. Ensure the response is well-structured and follows the format of the examples.
"""


def generate_user_content(religions_batch):
    """Formats the list of religions for the user message."""
    return "Religions: " + " \n ".join(religions_batch)


# --- MAIN GENERATION LOOP ---


def generate_dataset(num_files_to_generate=1000, batch_size=5):
    """
    Main function to generate the dataset.
    It takes a random sample of religions and appends each result to a single .jsonl file.
    """
    if len(MASTER_RELIGION_LIST) < num_files_to_generate * batch_size:
        print(
            "Warning: Master list is not large enough for the desired number of unique prompts."
        )
        return

    religion_pool = MASTER_RELIGION_LIST.copy()
    random.shuffle(religion_pool)

    with open(OUTPUT_FILENAME, "a", encoding="utf-8") as f:
        for i in range(num_files_to_generate):
            print(f"Generating entry {i+1}/{num_files_to_generate}...")

            if len(religion_pool) < batch_size:
                print("Not enough unique religions left in the pool. Stopping.")
                break

            religions_batch = [religion_pool.pop() for _ in range(batch_size)]

            user_content = generate_user_content(religions_batch)
            system_prompt = generate_system_prompt()

            try:
                # 1. Call the LLM API
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
                assistant_response = completion.choices[0].message.content

                # 2. Structure the data
                finetuning_pair = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_response},
                    ]
                }

                # 3. Convert to JSON string and append to file
                json_line = json.dumps(finetuning_pair, ensure_ascii=False)
                f.write(json_line + "\n")

                print(f"Successfully wrote entry {i+1} to {OUTPUT_FILENAME}")

            except Exception as e:
                print(f"An error occurred while generating entry {i+1}: {e}")
                time.sleep(5)
                religion_pool.extend(religions_batch)
                random.shuffle(religion_pool)


if __name__ == "__main__":
    generate_dataset(num_files_to_generate=587, batch_size=10)
