import os
import json
import random
import time
from openai import OpenAI
from duckduckgo_search import DDGS

# --- CONFIGURATION ---

# 1. Client for local Ollama model
ollama_client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

# 2. List of topics
MASTER_RELIGION_LIST = []
with open("all_rels_da.txt", "r", encoding="utf-8") as f:
    MASTER_RELIGION_LIST = f.readlines()
MASTER_RELIGION_LIST = [line.strip() for line in MASTER_RELIGION_LIST]

# 3. Output file
OUTPUT_FILENAME = "finetuning_dataset_ddgs.jsonl"


# --- HELPER FUNCTIONS ---

def generate_augmented_prompt(religions_batch, aggregated_context):
    """
    Creates the detailed prompt for the LLM, including the retrieved context for the whole batch.
    """
    system_prompt = """
You are a religious study scholar. Your task is to analyze a batch of several religions based *only* on the aggregated research context provided.
For each religion in the list, you must:
1. Write a short descriptive paragraph with information on the Metaphysics, Cosmology, Anthropology, Soteriology, Ethics & Morality, Praxis, and Epistemology of the each religion.
2. Create a markdown table with scores for Metaphysics, Cosmology, Anthropology, Soteriology, Ethics & Morality, Praxis, and Epistemology.

After analyzing all religions in the batch, you must:
3. Create a single, final markdown distance matrix table comparing all religions in the batch with each other.

Provide the full text output without the read more option.
"""
    
    religion_list_str = ", ".join(religions_batch)
    user_prompt = f"""
Please perform a full analysis for the following batch of religions: {religion_list_str}

--- AGGREGATED RESEARCH CONTEXT ---
{aggregated_context}
--- END OF CONTEXT ---

Based *only* on the context above, provide a single, complete response that analyzes all religions in the batch and concludes with the final distance matrix.
"""
    return system_prompt, user_prompt


# --- MAIN GENERATION LOOP ---

def generate_dataset(num_prompts_to_generate=200, batch_size=5):
    """
    Generates a dataset by creating batches of 5, searching the web for context for each,
    and then calling the local LLM once for the entire batch.
    """
    religion_pool = MASTER_RELIGION_LIST.copy()
    random.shuffle(religion_pool)

    # The DDGS object is used within a 'with' statement for proper session management.
    with open(OUTPUT_FILENAME, 'a', encoding='utf-8') as f, DDGS() as ddgs:
        for i in range(num_prompts_to_generate):
            print(f"--- Generating Batch {i+1}/{num_prompts_to_generate} ---")

            if len(religion_pool) < batch_size:
                print(f"Not enough unique religions left for a full batch of {batch_size}. Stopping.")
                break
            
            # 1. BATCHING: Get a batch of 5 religions
            religions_batch = [religion_pool.pop() for _ in range(batch_size)]
            print(f"  -> Batch: {', '.join(religions_batch)}")
            
            aggregated_context = ""
            try:
                # 2. SEARCH & AGGREGATE: Loop through the batch to gather context for each religion
                for religion in religions_batch:
                    print(f"    -> Searching for '{religion}'...")
                    search_query = (
                        f"Detailed analysis of the religion or philosophy of {religion}, "
                        f"including its Metaphysics, Cosmology, Anthropology, Soteriology, Ethics & Morality, Praxis, and Epistemology."
                    )
                    search_results = ddgs.text(search_query, max_results=3)
                    
                    context_for_one_religion = "\n".join([result['body'] for result in search_results])
                    
                    # Add structured headers to the aggregated context
                    aggregated_context += f"--- Context for {religion} ---\n{context_for_one_religion}\n\n"

                if not aggregated_context.strip():
                    print("  -> No search results found for the entire batch. Skipping.")
                    continue

                # 3. AUGMENT & GENERATE: Create one large prompt and call the local LLM
                print(f"  -> Generating analysis for the batch with Llama3...")
                system_prompt, user_prompt = generate_augmented_prompt(religions_batch, aggregated_context)
                
                completion = ollama_client.chat.completions.create(
                    model="llama3:8b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7 
                )
                assistant_response = completion.choices[0].message.content

                # 4. SAVE: Structure and save the entire batch interaction as one line
                finetuning_pair = {
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_response}
                    ]
                }
                json_line = json.dumps(finetuning_pair, ensure_ascii=False)
                f.write(json_line + '\n')
                
                print(f"  -> Successfully wrote batch {i+1} to {OUTPUT_FILENAME}\n")

            except Exception as e:
                print(f"An error occurred while processing batch {i+1}: {e}")
                # Add the failed batch back to the pool to retry later
                religion_pool.extend(religions_batch)
                random.shuffle(religion_pool)
                time.sleep(5)


# --- EXECUTE SCRIPT ---
if __name__ == "__main__":
    # This will generate 4 prompts, each containing a batch of 5 religions,
    # for a total of 20 religions processed. Adjust numbers as needed.
    generate_dataset(num_prompts_to_generate=1000, batch_size=5)