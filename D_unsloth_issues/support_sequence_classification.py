# sequence_classification.py

from unsloth import FastLanguageModel
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
import torch

def patch_sequence_classification(model_name, num_labels, id2label=None, label2id=None, load_in_4bit=True):
    """
    Patch AutoModelForSequenceClassification with Unsloth optimizations
    Args:
        model_name: Pretrained model name/path
        num_labels: Number of classification labels
        id2label: Optional label mapping
        label2id: Optional reverse label mapping
        load_in_4bit: Use 4-bit quantization
    Returns:
        Patched model and tokenizer
    """
    # Load base model with Unsloth optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=load_in_4bit,
        # Disable default causal LM head
        pretraining_model=False,
    )

    # Patch model class
    class UnslothSequenceClassification(model.__class__):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Add classification head
            self.score = torch.nn.Linear(
                self.config.hidden_size,
                num_labels,
                bias=self.config.problem_type == "regression"
            )
            
            # Initialize classifier
            self._init_weights(self.score)
            
            # Store label mappings
            self.config.id2label = id2label or {i: f"label_{i}" for i in range(num_labels)}
            self.config.label2id = label2id or {v: k for k, v in self.config.id2label.items()}

        def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            sequence_output = outputs[0][:, 0, :]  # Use [CLS] token
            logits = self.score(sequence_output)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    self.config.problem_type = "single_label_classification"
                    
                loss_fct = {
                    "regression": torch.nn.MSELoss(),
                    "single_label_classification": torch.nn.CrossEntropyLoss(),
                    "multi_label_classification": torch.nn.BCEWithLogitsLoss(),
                }[self.config.problem_type]
                
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return (loss, logits) if loss is not None else logits

    # Convert model to sequence classification
    model.__class__ = UnslothSequenceClassification
    model.num_labels = num_labels
    model.config.problem_type = "single_label_classification"

    # Add LoRA support
    lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )
    
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    return model, tokenizer

# Usage example
if __name__ == "__main__":
    model, tokenizer = patch_sequence_classification(
        model_name="unsloth/bert-base-uncased",
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
    )

    # Sample training setup
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        optim="adamw_torch",
        learning_rate=2e-4,
        num_train_epochs=3,
    )

    # Dummy dataset
    train_dataset = [
        {"text": "This movie was great!", "label": 1},
        {"text": "Terrible experience.", "label": 0},
    ]

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    tokenized_dataset = [tokenize_fn(ex) for ex in train_dataset]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    
    
#Create Patched Model

model, tokenizer = patch_sequence_classification(
    model_name="unsloth/llama-2-7b-bnb-4bit",
    num_labels=3,
)