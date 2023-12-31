# Automated Program Repair in the Era of Large Pre-trained Language Models

This repo contains both the patches generated by our study along with the code used to run the experiment

### Patch Generation

You can find the main patch generation logic in `Repair` folder

#### Local model generations
Example usage to run repair:
```
python repair --model_name EleutherAI/gpt-neo-1.3B \
              --batch_size 10 \
              --dataset defects4j \
              --chances 100 \
              --skip_val \
              --folder Results/test_generation \
```

#### Codex model generations
place your OpenAI access key in `Repair/Codex/api_key.txt`
```
python codex_repair --chances 100  \
                    --skip_val \
                    --folder Results/test_generation \
```

### Patch Validation

The validation function can be found in the `Dataset` folder with the appropriate function to validate each of the repair datasets

Example usage: 

first call the validation function with folder name and file name
`validate_all_patches(FOLDER_NAME, "lm_repair.json")` 

then run
`python validate_quixbug.py`

Note for ManyBugs Validation you first have to run the bugzoo server, see this for more detail: https://github.com/squaresLab/BugZoo
