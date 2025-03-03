## File Structure

Model_DPOTrainer.ipynb -> Selected a publicly available dataset, and implemented the Direct Preference Optimization (DPO) training method with DPOTrainer Function
using a pre-trained transformer model (GPT2). Save the trained model.

Hugging_Face.ipynb -> Loaded the Saved trained model and uploaded the model to the Hugging Face Model Hub.

app.py -> The app allow users to input text and receive response.

## Experiment (hyperparameters and training performance)

Hyperparameters

    num_train_epochs=3,
    learning_rate=5e-07,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    do_eval=True,
    per_device_eval_batch_size=1,
    adam_epsilon=1e-08,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    seed=42,
    logging_steps=100,
    save_steps=500,
    save_strategy="steps",
    output_dir="./output-dir",
    gradient_checkpointing=True,
    bf16=False,  # Disable bf16 for CPU
    remove_unused_columns=False,
    beta=0.1,
    max_length=512,
    max_prompt_length=256
    
Training performance

|Step	|Training Loss |
|-----|--------------|
|100	|0.694500 |
|200	|0.688000 |
|300	|0.688300 |
|400	|0.696100 |
|500	|0.689300 |
Note: I used 1% of the dataset for this training, and it took 5 hours, 12 minutes, and 53 seconds on a CPU.

Conclusions

1. The current training setup is not effective, as indicated by the stagnant training loss.

2. The primary issues are likely the low learning rate, small dataset size, and insufficient training epochs.

## Link for Hugging Face Hub

https://huggingface.co/arunyasenadeera/A5dpo-finetuned-model

## Web Application Development
[![Watch the video](https://img.youtube.com/vi/8nl4b9R7dkQ/0.jpg)](https://www.youtube.com/watch?v=8nl4b9R7dkQ)
