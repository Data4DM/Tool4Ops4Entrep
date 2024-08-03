import os
import openai

# Set up your OpenAI API key
openai.api_key = 'your key'

def process_paper(paper_content):
    prompt = (f"Imagine you're the author of this paper. Please choose multiple hypotheses (among UE1,2,3,4, UI1,2,3,4) "
              "that the author of each paper would support and add one sentence reasoning for that support.\n\n"
              f"Paper Content:\n{paper_content}\n\n")
    
    response = openai.Completion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    return response.choices[0].message['content'].strip()

def main(folder_path):
    results = {}

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            try:
                # Try reading the file with 'utf-8' encoding
                with open(file_path, 'r', encoding='utf-8') as f:
                    paper_content = f.read()
            except UnicodeDecodeError:
                # If that fails, use 'latin1' encoding as a fallback
                with open(file_path, 'r', encoding='latin1') as f:
                    paper_content = f.read()

            # Process the paper content with GPT API
            analysis = process_paper(paper_content)

            # Save the result
            results[filename] = analysis

    # Output the results
    for filename, analysis in results.items():
        print(f"{filename}:\n{analysis}\n")

if __name__ == '__main__':
    folder_path = '../db/db(bayes(experiment))'
    main(folder_path)
