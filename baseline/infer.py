import argparse
import os

import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
from tqdm import tqdm

from paddlenlp.transformers import AutoTokenizer, AutoModel


# ==========================================
# 1. Model  (must match train.py)
# ==========================================
class CPAModel(nn.Layer):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)

        hidden_size = None
        if hasattr(self.encoder, 'config'):
            if isinstance(self.encoder.config, dict):
                hidden_size = self.encoder.config.get('hidden_size', None)
            else:
                hidden_size = getattr(self.encoder.config, 'hidden_size', None)
        if hidden_size is None and hasattr(self.encoder, 'embeddings') and hasattr(self.encoder.embeddings, 'word_embeddings'):
            hidden_size = self.encoder.embeddings.word_embeddings.weight.shape[-1]
        if hidden_size is None:
            raise ValueError('Unable to infer hidden_size automatically.')

        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.classifier = nn.Linear(hidden_size, num_labels)
        self.rel_prototypes = paddle.create_parameter(
            shape=[num_labels, hidden_size], dtype='float32',
            default_initializer=nn.initializer.XavierUniform()
        )
        self.temperature = paddle.create_parameter(
            shape=[1], dtype='float32',
            default_initializer=nn.initializer.Constant(0.05)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, tuple):
            seq_out = outputs[0]
        elif hasattr(outputs, 'last_hidden_state'):
            seq_out = outputs.last_hidden_state
        else:
            seq_out = outputs

        cls_emb = seq_out[:, 0, :]

        cls_logits = self.classifier(self.dropout(cls_emb))
        cls_norm = F.normalize(cls_emb, axis=-1)
        proto_norm = F.normalize(self.rel_prototypes, axis=-1)
        cont_logits = cls_norm @ proto_norm.T / self.temperature

        return cls_logits, cont_logits


# ==========================================
# 2. Tokenization  (matches train.py)
# ==========================================
def encode_text_pair(tokenizer, text_a, text_b, max_length):
    try:
        encoding = tokenizer(
            text=text_a, text_pair=text_b,
            max_length=max_length, padding='max_length',
            truncation=True, return_attention_mask=True,
        )
    except TypeError:
        try:
            encoding = tokenizer(
                text_a, text_b,
                max_length=max_length, padding='max_length',
                truncation=True, return_attention_mask=True,
            )
        except TypeError:
            encoding = tokenizer(
                text=text_a, text_pair=text_b,
                max_seq_len=max_length, pad_to_max_seq_len=True,
                truncation=True, return_attention_mask=True,
            )

    input_ids = encoding['input_ids']
    attention_mask = encoding.get('attention_mask', None)
    if attention_mask is None:
        seq_len = encoding.get('seq_len', len(input_ids))
        attention_mask = [1] * min(seq_len, max_length) + [0] * (max_length - min(seq_len, max_length))

    return np.array(input_ids, dtype='int64'), np.array(attention_mask, dtype='int64')


# ==========================================
# 3. Inference dataset
# ==========================================
class InferenceDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.original_rows = []

        df = pd.read_csv(csv_path, low_memory=False, encoding='utf-8-sig')
        df.columns = [str(col).strip() for col in df.columns]

        subject_col = object_col = None
        for col in df.columns:
            if col.lower() == 'subject':
                subject_col = col
            elif col.lower() == 'object':
                object_col = col

        if subject_col is None or object_col is None:
            raise ValueError("CSV must have 'Subject' and 'Object' columns.")

        temp_df = df[[subject_col, object_col]].dropna()
        for idx, row in temp_df.iterrows():
            self.samples.append((str(row[subject_col]), str(row[object_col])))
            self.original_rows.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subj, obj = self.samples[idx]
        input_ids, attention_mask = encode_text_pair(
            self.tokenizer, subj, obj, self.max_length
        )
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'orig_idx': np.int64(idx),
        }


def collate_fn(samples):
    return {
        'input_ids': np.stack([s['input_ids'] for s in samples]).astype('int64'),
        'attention_mask': np.stack([s['attention_mask'] for s in samples]).astype('int64'),
        'orig_idx': np.array([s['orig_idx'] for s in samples], dtype='int64'),
    }


# ==========================================
# 4. Single model inference
# ==========================================
def predict_with_model(model, dataloader, use_amp, device, contrastive_beta=0.5):
    """Run inference with a single model, returning logits for each sample."""
    model.eval()
    all_logits = []

    with paddle.no_grad():
        for batch in tqdm(dataloader, desc='Inference', leave=False):
            ids = paddle.to_tensor(batch['input_ids'], dtype='int64')
            mask = paddle.to_tensor(batch['attention_mask'], dtype='int64')

            if use_amp:
                with paddle.amp.auto_cast(enable=True):
                    cls_logits, cont_logits = model(ids, mask)
            else:
                cls_logits, cont_logits = model(ids, mask)

            final = cls_logits + contrastive_beta * cont_logits
            all_logits.append(final.numpy())

    return np.concatenate(all_logits, axis=0)  # (N, C)


def load_model(model_path, shortcut_name, num_classes):
    """Load a single trained model."""
    model = CPAModel(shortcut_name, num_classes)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')
    state_dict = paddle.load(model_path)
    model.set_state_dict(state_dict)
    return model


# ==========================================
# 5. Device helper
# ==========================================
def resolve_device(device_arg):
    try:
        custom_types = paddle.device.get_all_custom_device_type()
    except Exception:
        custom_types = []
    print(f'Available custom devices: {custom_types}')
    if device_arg:
        try:
            dev = paddle.set_device(device_arg)
            print(f'Using: {dev}')
            return dev
        except Exception as e:
            print(f'Failed: {e}')
    dev = paddle.set_device('cpu')
    print('Falling back to CPU.')
    return dev


# ==========================================
# 6. Inference pipeline
# ==========================================
def run_inference(args):
    device = resolve_device(args.device)

    # Load label mapping
    with open(args.labels_path, 'r', encoding='utf-8-sig') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    id2label = {i: label for i, label in enumerate(classes)}
    print(f'Loaded {len(classes)} labels.')

    # Parse model paths (support ensemble: comma-separated)
    model_paths = [p.strip() for p in args.model_path.split(',') if p.strip()]
    print(f'Models to ensemble: {len(model_paths)}')

    # Tokenizer (once)
    tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)

    # Dataset & DataLoader
    dataset = InferenceDataset(args.input_csv, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, return_list=True,
    )
    print(f'Total valid rows: {len(dataset)}')

    use_amp = args.use_amp and str(device) != 'cpu'

    # Ensemble: average logits across all models
    sum_logits = None
    for i, mp in enumerate(model_paths):
        print(f'Loading model {i + 1}/{len(model_paths)}: {mp}')
        model = load_model(mp, args.shortcut_name, len(classes))
        logits = predict_with_model(model, dataloader, use_amp, device, args.contrastive_beta)
        if sum_logits is None:
            sum_logits = logits.astype(np.float64)
        else:
            sum_logits += logits.astype(np.float64)

    avg_logits = sum_logits / len(model_paths)
    pred_indices = np.argmax(avg_logits, axis=1).tolist()

    # Build predictions
    predictions = [id2label[idx] for idx in pred_indices]

    # Attach to original CSV
    original_df = pd.read_csv(args.input_csv, low_memory=False, encoding='utf-8-sig')
    original_df.columns = [str(col).strip() for col in original_df.columns]

    subject_col = object_col = None
    for col in original_df.columns:
        if col.lower() == 'subject':
            subject_col = col
        elif col.lower() == 'object':
            object_col = col

    valid_mask = original_df[subject_col].notna() & original_df[object_col].notna()
    valid_indices = original_df[valid_mask].index.tolist()

    original_df['Label'] = None
    for row_idx, pred_label in zip(valid_indices, predictions):
        original_df.loc[row_idx, 'Label'] = pred_label

    out_df = original_df[[subject_col, object_col, 'Label']]
    out_df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
    print(f'Done. Saved to: {args.output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default="./dataset/test.csv")
    parser.add_argument('--labels_path', type=str, default="./dataset/labels.txt")
    # Comma-separated model paths for ensemble
    parser.add_argument('--model_path', type=str,
                        default="./cpa_output/cpa_20260521_150928/best_model.pdparams")
    parser.add_argument('--output_file', type=str, default='./submission.csv')
    parser.add_argument('--shortcut_name', type=str, default='roberta-base')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--contrastive_beta', type=float, default=0.5,
                        help='Weight for contrastive logits at inference')
    args = parser.parse_args()
    run_inference(args)
