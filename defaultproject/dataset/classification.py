import os
import datasets
import itertools
import functools
import torch

def prepare_classification_dataset(tokenizer,
                             dataset_name_or_path: str = "SetFit/20_newsgroups",
                             dataset_subset: str = None,
                             text_column_name: str = "text",
                             label_column_name: str = "label",
                             train_sample_size: int =-1,
                             cache_path:str="cache"):
    if cache_path is not None and os.path.exists(os.path.join(cache_path,"classification")):
        print(f"Using pre-downloaded dataset from {cache_path}.")
        train = datasets.load_from_disk(os.path.join(cache_path,"classification","train"))
        validation = datasets.load_from_disk(os.path.join(cache_path,"classification","eval"))
        return train, validation
    else:
        print(f"Download and prerpocessing dataset from {dataset_name_or_path} on subset {dataset_subset}...")
        dataset = datasets.load_dataset(dataset_name_or_path, dataset_subset)
        if "eval" in dataset:
            validation_split = "eval"
        elif "test" in dataset:
            validation_split = "test"
        else:
            print("Using train split for validation.")
            dataset = datasets.train_test_split(test_size=0.1)
            validation_split = "test"
        
        train = dataset["train"] if train_sample_size == -1 else dataset["train"].select(range(train_sample_size))
        validation = dataset[validation_split]

        processing_lambda = functools.partial(
            _preprocessing,
            tokenizer=tokenizer,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
        )

        train = train.map(
            processing_lambda,
            batched=True,
            remove_columns=train.column_names,
            num_proc=4,
            desc="Tokenizing",
        )
        validation = validation.map(
            processing_lambda,
            batched=True,
            remove_columns=validation.column_names,
            num_proc=4,
            desc="Tokenizing",
        )

        if cache_path is not None:
            os.makedirs(os.path.join(cache_path,"classification"), exist_ok=True)
            print(f"Saving dataset to {cache_path}...")
            train.save_to_disk(os.path.join(cache_path,"classification","train"), max_shard_size="500MB")
            validation.save_to_disk(os.path.join(cache_path,"classification","eval"), max_shard_size="500MB")
        return train, validation

def _preprocessing(examples, tokenizer, text_column_name, label_column_name):
    text = examples[text_column_name]
    labels = examples[label_column_name]

    inputs = tokenizer(text, add_special_tokens=True, padding="longest", return_tensors="pt",max_length=512, truncation=True)
    inputs["labels"] = torch.tensor(labels, dtype=torch.long)

    return inputs

def collate_fn_for_classification(batch, tokenizer,):
    # fill here
    ######
    
    #### YOUR CODES; TODO 
    # drop the padding added per map-batch in _preprocessing, then left-pad to the longest sequence of this batch
    # (left padding: TransformerForSequenceClassification classifies with the hidden state of the last position)
    sequences = [
        [token for token, mask in zip(example["input_ids"], example["attention_mask"]) if mask == 1]
        for example in batch
    ]
    max_length = max(max(len(seq) for seq in sequences), 1)
    input_ids = torch.full((len(batch), max_length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
    for i, seq in enumerate(sequences):
        if len(seq) > 0:
            input_ids[i, -len(seq):] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, -len(seq):] = 1
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor([example["labels"] for example in batch], dtype=torch.long),
    }
    ######
    return inputs