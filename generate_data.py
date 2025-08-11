import os
import json
import random
from openai import OpenAI

# --- CONFIGURATION ---

# 1. Set up your API key.
# It's best to set this as an environment variable for security.
# client = OpenAI(api_key="YOUR_API_KEY") 
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 2. Populate this list with the 5000 religions/philosophies you want to process.
# This is a small sample to show the structure.
MASTER_RELIGION_LIST = []
with open("all_rels_da.txt", "r", encoding="utf-8") as f:
    MASTER_RELIGION_LIST = f.readlines()
MASTER_RELIGION_LIST = [line.strip() for line in MASTER_RELIGION_LIST]

# 3. Define the output directory.
OUTPUT_DIR = "finetuning_dataset"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- HELPER FUNCTIONS ---

def generate_system_prompt():
    """Defines the core instruction for the language model."""
    # This is the detailed instruction set from your original prompt.
    return """
Role: Religious study scholar. Give me a short descriptive paragraph of the following religions. Provide information on the dimensions Metaphysics, Cosmology, Anthropology, Soteriology, Ethics & Morality, Praxis, and Epistemology for each religion. Provide a score for each dimension for each religion and calculate the distance between each religion based on the scores.

The scoring rubric for each dimension is on a 1-5 scale:
* Metaphysics: 1 (Purely Materialist) → 5 (Strongly Transcendent/Supernatural)
* Cosmology: 1 (Purely Naturalistic) → 5 (Direct Supernatural Creation)
* Anthropology: 1 (Purely Biological) → 5 (Possessing a Divine/Eternal Soul)
* Soteriology: 1 (This-Worldly/Self-Achieved) → 5 (Requires Divine Grace/Intervention)
* Ethics: 1 (Human-Derived/Situational) → 5 (Divinely Commanded/Absolute)
* Praxis: 1 (Minimal/No Ritual) → 5 (Highly Prescribed Rituals)
* Epistemology: 1 (Purely Empirical/Rational) → 5 (Primarily based on Divine Revelation)

Generate a response that includes a descriptive paragraph for each religion, a markdown table with scores for each dimension, and a final markdown table with the calculated Euclidean distance matrix between the religions. Ensure the response is well-structured and follows the format of the examples.
"""

def generate_user_content(religions_batch):
    """Formats the list of religions for the user message."""
    return "Religions: " + " \n ".join(religions_batch)

# --- MAIN GENERATION LOOP ---

def generate_dataset(num_files_to_generate=1000, batch_size=5):
    """
    Main function to generate the dataset.
    It takes a random sample of religions and generates a file for each batch.
    """
    if len(MASTER_RELIGION_LIST) < num_files_to_generate * batch_size:
        print("Warning: Master list is not large enough for the desired number of unique files.")
        return

    # Use a copy to avoid modifying the original list
    religion_pool = MASTER_RELIGION_LIST.copy()
    random.shuffle(religion_pool)

    for i in range(num_files_to_generate):
        print(f"Generating file {i+1}/{num_files_to_generate}...")

        # 1. Get a batch of religions
        if len(religion_pool) < batch_size:
            print("Not enough unique religions left in the pool. Stopping.")
            break
        
        religions_batch = [religion_pool.pop() for _ in range(batch_size)]
        
        user_content = generate_user_content(religions_batch)
        system_prompt = generate_system_prompt()

        try:
            # 2. Call the LLM API
            completion = client.chat.completions.create(
                model="gpt-4o-mini", # Or your model of choice
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
            )
            assistant_response = completion.choices[0].message.content

            # 3. Structure the data in the desired JSON format
            finetuning_pair = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_response}
                ]
            }

            # 4. Save the file
            file_path = os.path.join(OUTPUT_DIR, f"religion_data_{i+1}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(finetuning_pair, f, indent=4, ensure_ascii=False)
            
            print(f"Successfully saved {file_path}")

        except Exception as e:
            print(f"An error occurred while generating file {i+1}: {e}")
            # Optional: Add the failed batch back to the pool to retry later
            religion_pool.extend(religions_batch)

# --- EXECUTE SCRIPT ---
if __name__ == "__main__":
    # This will generate 1000 files, each containing 5 religions.
    generate_dataset(num_files_to_generate=1000, batch_size=5)