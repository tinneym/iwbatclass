from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer, TrainerCallback
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import evaluate
import librosa
import torchaudio
import torch
from sklearn.preprocessing import LabelEncoder

#-----------------------------------------------------------------------------------
# FILE FOR APPLYING CLASS WEIGHTING
#-----------------------------------------------------------------------------------

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load audio data
dataset = load_dataset("audiofolder", data_dir="/n/fs/iwbatclass/audio/train", split='train')
train_test_split = dataset.train_test_split(test_size=0.2)

small_train_dataset = train_test_split['train']
small_eval_dataset = train_test_split['test']

# Set original and new sampling rates
original_sampling_rate = 250000  
target_sampling_rate = 16000    

# Initialize feature extractor and model
config = AutoConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
config.num_labels = 13 
feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", 
                                                        config=config,
                                                        ignore_mismatched_sizes=True)
model.to(device) 

# Determine class weights from the training dataset
labels = np.array(small_train_dataset['label'])  
class_counts = np.bincount(labels, minlength=13)  
class_weights = compute_class_weight('balanced', classes=np.arange(13), y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Function to extract features
def extract_features(batch):
    batch_features = []
    resampler = torchaudio.transforms.Resample(original_sampling_rate, target_sampling_rate).to(device)

    for audio in batch["audio"]:
        audio_array = torch.tensor(audio["array"], dtype=torch.float32).to(device)  
        resampled_audio = resampler(audio_array) 

        # Move resampled audio to CPU for feature extraction
        resampled_audio = resampled_audio.cpu()
        
        # Execute feature extraction
        features = feature_extractor(resampled_audio.numpy(), sampling_rate=target_sampling_rate, return_tensors="np")
        batch_features.append(features.input_values[0])

    return {"input_values": batch_features}

# Apply feature extraction to the training and evaluation datasets
small_train_dataset = small_train_dataset.map(extract_features, batched=True, batch_size=16)
small_eval_dataset = small_eval_dataset.map(extract_features, batched=True, batch_size=16)

# Set hyperparameters to tune and evaluation metrics
training_args = TrainingArguments(
    output_dir="/n/fs/iwbatclass/test_trainer", 
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,  
    save_strategy="no",
    logging_dir=None,  
    logging_strategy="no",
    logging_steps=0,
    logging_first_step=False
)

# Load evaluation metrics
accuracy_metric = evaluate.load("accuracy")
recall_metric = evaluate.load("recall")
precision_metric = evaluate.load("precision")
f1_metric = evaluate.load("f1")
confusion_metric = evaluate.load("confusion_matrix")

# Function to convert logits to predictions for each metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(predictions=predictions, references=labels, average='macro', zero_division = 1)
    precision = precision_metric.compute(predictions=predictions, references=labels, average='macro', zero_division = 1)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')
    confusion_matrix = confusion_metric.compute(predictions=predictions, references=labels)
    return {
        'accuracy': accuracy['accuracy'],
        'recall': recall['recall'],
        'precision': precision['precision'],
        'f1': f1['f1'],
        'confusion matrix': confusion_matrix['confusion_matrix']
    }

# Function for printing out the epoch number
class PrintEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} finished.")

# Function for creating a custom Trainer to deal with class weighting
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Create a Trainer object
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[PrintEpochCallback()]
)

# Train the model
trainer.train()
