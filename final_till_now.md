```python
!pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
!pip install -q datasets bitsandbytes einops wandb
!pip install gradientai --upgrade
import os
os.environ['GRADIENT_ACCESS_TOKEN'] = "RrJ8RaxPoDEY8tteUdCjkIPmnPfj2NO5"
os.environ['GRADIENT_WORKSPACE_ID'] = "f0e45fb4-a474-48ed-97b9-dd2cde73cbd5_workspace"
```

      Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m225.0/225.0 kB[0m [31m3.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m8.8/8.8 MB[0m [31m16.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m290.1/290.1 kB[0m [31m28.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m510.5/510.5 kB[0m [31m26.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m79.8/79.8 kB[0m [31m9.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m23.7/23.7 MB[0m [31m42.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m823.6/823.6 kB[0m [31m52.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m14.1/14.1 MB[0m [31m64.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m731.7/731.7 MB[0m [31m930.6 kB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m410.6/410.6 MB[0m [31m3.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m121.6/121.6 MB[0m [31m8.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m56.5/56.5 MB[0m [31m10.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m124.2/124.2 MB[0m [31m6.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m196.0/196.0 MB[0m [31m2.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m166.0/166.0 MB[0m [31m2.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m99.1/99.1 kB[0m [31m12.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.1/21.1 MB[0m [31m61.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m116.3/116.3 kB[0m [31m6.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m194.1/194.1 kB[0m [31m19.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m134.8/134.8 kB[0m [31m15.2 MB/s[0m eta [36m0:00:00[0m
    [?25h  Building wheel for peft (pyproject.toml) ... [?25l[?25hdone
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m102.2/102.2 MB[0m [31m8.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m44.6/44.6 kB[0m [31m5.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.2/2.2 MB[0m [31m72.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m195.4/195.4 kB[0m [31m22.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m264.9/264.9 kB[0m [31m28.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m62.7/62.7 kB[0m [31m8.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting gradientai
      Downloading gradientai-1.8.1-py3-none-any.whl (296 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m296.5/296.5 kB[0m [31m4.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting aenum>=3.1.11 (from gradientai)
      Downloading aenum-3.1.15-py3-none-any.whl (137 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m137.6/137.6 kB[0m [31m10.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting pydantic<2.0.0,>=1.10.5 (from gradientai)
      Downloading pydantic-1.10.14-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.1/3.1 MB[0m [31m12.5 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from gradientai) (2.8.2)
    Requirement already satisfied: urllib3>=1.25.3 in /usr/local/lib/python3.10/dist-packages (from gradientai) (2.0.7)
    Requirement already satisfied: typing-extensions>=4.2.0 in /usr/local/lib/python3.10/dist-packages (from pydantic<2.0.0,>=1.10.5->gradientai) (4.10.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->gradientai) (1.16.0)
    Installing collected packages: aenum, pydantic, gradientai
      Attempting uninstall: pydantic
        Found existing installation: pydantic 2.6.4
        Uninstalling pydantic-2.6.4:
          Successfully uninstalled pydantic-2.6.4
    Successfully installed aenum-3.1.15 gradientai-1.8.1 pydantic-1.10.14



```python
!huggingface-cli login
```

    
        _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
        _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
        _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
        _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
        _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|
    
        To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
    Token: 
    Add token as git credential? (Y/n) Y
    Token is valid (permission: write).
    [1m[31mCannot authenticate through git-credential as no helper is defined on your machine.
    You might have to re-authenticate when pushing to the Hugging Face Hub.
    Run the following command in your terminal in case you want to set the 'store' credential helper as default.
    
    git config --global credential.helper store
    
    Read https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage for more details.[0m
    Token has not been saved to git credential helper.
    Your token has been saved to /root/.cache/huggingface/token
    Login successful



```python
import re
import json
from datasets import load_dataset
from gradientai import Gradient
```


```python
def main():
  with Gradient() as gradient:
      base_model = gradient.get_base_model(base_model_slug="nous-hermes2")

      new_model_adapter = base_model.create_model_adapter(
          name="test model 3"
      )


      def count_tokens(response):
       tokens = response.split()  # Splitting by whitespace
       return len(tokens)

      dataset = load_dataset('timdettmers/openassistant-guanaco')


      # Shuffle the dataset and slice it
      dataset = dataset['train'].shuffle(seed=42).select(range(1000))

      # Define a function to transform the data
      def transform_conversation(example):
          conversation_text = example['text']
          segments = conversation_text.split('###')

          reformatted_segments = []

          # Iterate over pairs of segments
          for i in range(1, len(segments) - 1, 2):
              human_text = segments[i].strip().replace('Human:', '').strip()

              # Check if there is a corresponding assistant segment before processing
              if i + 1 < len(segments):
                  assistant_text = segments[i+1].strip().replace('Assistant:', '').strip()

                  # Apply the new template
                  reformatted_segments.append(f'<s>[INST] {human_text} [/INST] {assistant_text} </s>')
              else:
                  # Handle the case where there is no corresponding assistant segment
                  reformatted_segments.append(f'<s>[INST] {human_text} [/INST] </s>')

          return {'text': ''.join(reformatted_segments)}


     # Apply the transformation
      transformed_dataset = dataset.map(transform_conversation) #NEW DATASET

      transformed_dataset.push_to_hub("guanaco-llama2-1k")      #PUSHED TO HF

      # Extract question-answer pairs
      question_answer_pairs = []
      for example in transformed_dataset:
          # Extract question and answer using regular expressions
          matches = re.findall(r'### Human: (.+?)### Assistant: (.+?)(?=(### Human:|$))', example['text'], re.DOTALL)
          # Append each question-answer pair to the list
          question_answer_pairs.extend(matches)

      # Format question-answer pairs into samples
      samples = [{"inputs": f"### Instruction: {pair[0].strip()} \n\n### Response: {pair[1].strip()}"} for pair in question_answer_pairs]

# Remove instructions with less than 100 tokens in the response
      filtered_dataset = []
      deduplicated_samples = []  # Define deduplicated_samples outside of the loop

      for item in samples:
          response = item["inputs"].split("### Response:")[1].strip()  # Extract response

          if count_tokens(response) >= 100:
              filtered_dataset.append(item)
              # Data deduplication using cosine similarity
              for new_sample in filtered_dataset:
                  is_unique = True
                  new_response = new_sample["inputs"].split("### Response:")[1].strip()
                  for existing_sample in deduplicated_samples:
                      existing_response = existing_sample["inputs"].split("### Response:")[1].strip()
                      similarity = cosine_sim(new_response, existing_response)
                      if similarity > 0.95:
                          is_unique = False
                          break  # Exit the inner loop if similarity > 0.95

                  if is_unique:
                      deduplicated_samples.append(new_sample)

      # Print deduplicated samples
      print("Deduplicated samples:")
      for sample in deduplicated_samples:
          print(sample)

      # Fine-tuning the model
      sample_query = "### Instruction: I want to start doing astrophotography as a hobby, any suggestions what could i do? \n\n### Response:"
      num_epochs = 4
      count = 0
      while count < num_epochs:
          print(f"Fine-tuning the model, iteration {count + 1}")
          new_model_adapter.fine_tune(samples=samples)
          count = count + 1

      # After fine-tuning
      completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=50).generated_output
      print(f"Generated (after fine-tune): {completion}")

      new_model_adapter.delete()


if __name__ == "__main__":
      main()
```

    /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:88: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(



    Downloading readme:   0%|          | 0.00/395 [00:00<?, ?B/s]


    /usr/local/lib/python3.10/dist-packages/huggingface_hub/repocard.py:105: UserWarning: Repo card metadata block was not found. Setting CardData to empty.
      warnings.warn("Repo card metadata block was not found. Setting CardData to empty.")



    Downloading data:   0%|          | 0.00/20.9M [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/1.11M [00:00<?, ?B/s]



    Generating train split: 0 examples [00:00, ? examples/s]



    Generating test split: 0 examples [00:00, ? examples/s]



    Map:   0%|          | 0/1000 [00:00<?, ? examples/s]



    Uploading the dataset shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Creating parquet from Arrow format:   0%|          | 0/1 [00:00<?, ?ba/s]



    README.md:   0%|          | 0.00/273 [00:00<?, ?B/s]


    Deduplicated samples:
    Fine-tuning the model, iteration 1
    Fine-tuning the model, iteration 2
    Fine-tuning the model, iteration 3
    Fine-tuning the model, iteration 4
    Generated (after fine-tune):  Astrophotography can be a rewarding hobby that allows you to capture the beauty of the night sky. Here are some suggestions to help you get started:
    
    1. Invest in a sturdy tripod: A good

