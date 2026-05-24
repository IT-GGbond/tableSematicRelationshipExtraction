import argparse
import os
import random
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from paddlenlp.transformers import AutoTokenizer, AutoModel, LinearDecayWithWarmup

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_data_from_directory(dir_path):
    all_data = []
    if not os.path.exists(dir_path):
        raise ValueError(f"can't find: {dir_path}")

    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    logging.info(f"load data from {dir_path} ...")

    for filename in tqdm(csv_files, desc=f"loading {os.path.basename(dir_path)}"):
        file_path = os.path.join(dir_path, filename)
        label_name = filename[:-4]
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding='utf-8-sig')
            if df.empty:
                continue
            df.columns = [str(col).strip() for col in df.columns]
            if 'Subject' in df.columns and 'Object' in df.columns:
                df = df[['Subject', 'Object']].dropna()
                df['label'] = label_name
                all_data.append(df)
        except Exception as e:
            logging.warning(f"{filename} load error: {e}")

    if not all_data:
        raise ValueError(f"{dir_path} not valid data")

    full_df = pd.concat(all_data, ignore_index=True)
    full_df['Subject'] = full_df['Subject'].astype(str)
    full_df['Object'] = full_df['Object'].astype(str)
    return full_df


# ---------------------------------------------------------------------------
# 2. Text encoding — text-pair, consistent with infer.py
# ---------------------------------------------------------------------------
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


def encode_single_text(tokenizer, text, max_length=32):
    """Encode a single text (used for relation name initialization)."""
    try:
        encoding = tokenizer(
            text=text, max_length=max_length, padding='max_length',
            truncation=True, return_attention_mask=True,
        )
    except TypeError:
        encoding = tokenizer(
            text=text, max_seq_len=max_length, pad_to_max_seq_len=True,
            truncation=True, return_attention_mask=True,
        )
    input_ids = encoding['input_ids']
    mask = encoding.get('attention_mask', encoding.get('attention_mask', None))
    if mask is None:
        seq_len = encoding.get('seq_len', len(input_ids))
        mask = [1] * min(seq_len, max_length) + [0] * (max_length - min(seq_len, max_length))
    return np.array(input_ids, dtype='int64'), np.array(mask, dtype='int64')


# ---------------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------------
class RelationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_encoder, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.le = label_encoder
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_ids, attention_mask = encode_text_pair(
            self.tokenizer, row['Subject'], row['Object'], self.max_length
        )
        label_id = self.le.transform([row['label']])[0]
        return {
            'valid': True,
            'token_ids': input_ids,
            'cls_mask': attention_mask,
            'label_id': np.int64(label_id),
        }


def dynamic_collate_fn(samples):
    valid_samples = [s for s in samples if s.get('valid', False)]
    if not valid_samples:
        return None
    return {
        'data': np.stack([s['token_ids'] for s in valid_samples]).astype('int64'),
        'label': np.array([s['label_id'] for s in valid_samples], dtype='int64'),
        'cls_mask': np.stack([s['cls_mask'] for s in valid_samples]).astype('int64'),
    }


# ---------------------------------------------------------------------------
# 4. Model  — dual-head: classifier + contrastive prototypes
# ---------------------------------------------------------------------------
class CPAModel(nn.Layer):
    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        hidden_size = None
        if hasattr(self.encoder, 'config'):
            if isinstance(self.encoder.config, dict):
                hidden_size = self.encoder.config.get('hidden_size', None)
            else:
                hidden_size = getattr(self.encoder.config, 'hidden_size', None)
        if hidden_size is None and hasattr(self.encoder, 'embeddings') and hasattr(self.encoder.embeddings, 'word_embeddings'):
            hidden_size = self.encoder.embeddings.word_embeddings.weight.shape[-1]
        if hidden_size is None:
            raise ValueError('hidden_size is None')

        self.hidden_size = hidden_size
        self.num_labels = num_labels

        # Classification head
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Learnable relation prototypes for contrastive learning
        self.rel_prototypes = paddle.create_parameter(
            shape=[num_labels, hidden_size], dtype='float32',
            default_initializer=nn.initializer.XavierUniform()
        )
        # Fixed temperature (NOT learnable — avoids NaN from τ → 0)
        self.register_buffer('temperature', paddle.to_tensor([0.05], dtype='float32'))

    def _get_cls(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, tuple):
            seq_out = outputs[0]
        elif hasattr(outputs, 'last_hidden_state'):
            seq_out = outputs.last_hidden_state
        else:
            seq_out = outputs
        return seq_out[:, 0, :]

    def forward(self, input_ids, attention_mask):
        cls_emb = self._get_cls(input_ids, attention_mask)          # (B, H)

        # Classification logits
        cls_logits = self.classifier(self.dropout(cls_emb))         # (B, C)

        # Contrastive logits via dot-product with relation prototypes
        cls_norm = F.normalize(cls_emb, axis=-1)                    # (B, H)
        proto_norm = F.normalize(self.rel_prototypes, axis=-1)      # (C, H)
        cont_logits = cls_norm @ proto_norm.T / self.temperature    # (B, C)

        return cls_logits, cont_logits

    def init_prototypes_from_names(self, tokenizer, label_names, device='gpu'):
        """Pre-compute BERT encodings of relation names as prototype initial values."""
        logging.info('Initializing relation prototypes from BERT encodings...')
        bert = self.encoder
        proto_buf = []
        bs = 64
        for i in range(0, len(label_names), bs):
            batch_names = label_names[i:i + bs]
            ids_list, mask_list = [], []
            for name in batch_names:
                ids, mask = encode_single_text(tokenizer, name, 32)
                ids_list.append(ids)
                mask_list.append(mask)
            ids_t = paddle.to_tensor(np.stack(ids_list), dtype='int64')
            mask_t = paddle.to_tensor(np.stack(mask_list), dtype='int64')
            with paddle.no_grad():
                out = bert(input_ids=ids_t, attention_mask=mask_t)
                if isinstance(out, tuple):
                    out = out[0]
                elif hasattr(out, 'last_hidden_state'):
                    out = out.last_hidden_state
                proto_buf.append(out[:, 0, :].numpy())              # (bs, H)
        init_vals = np.concatenate(proto_buf, axis=0)               # (C, H)
        self.rel_prototypes.set_value(paddle.to_tensor(init_vals, dtype='float32'))
        logging.info('Prototypes initialized.')


# ---------------------------------------------------------------------------
# 5. Utilities
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(save_dir, 'train.log'), mode='w', encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )


def resolve_device(device_arg):
    try:
        custom_types = paddle.device.get_all_custom_device_type()
    except Exception:
        custom_types = []
    logging.info(f'custom device types: {custom_types}')
    if device_arg:
        try:
            dev = paddle.set_device(device_arg)
            logging.info(f'use: {dev}')
            return dev
        except Exception as e:
            logging.warning(f'{device_arg} use error: {e}')
    dev = paddle.set_device('cpu')
    logging.warning('set device to CPU')
    return dev


def save_label_classes(label_encoder, save_dir):
    path = os.path.join(save_dir, 'label_classes.txt')
    with open(path, 'w', encoding='utf-8') as f:
        for label in label_encoder.classes_:
            f.write(f'{label}\n')


def oversample_rare_classes(df, min_count=10, max_multiplier=10):
    counts = df['label'].value_counts()
    dfs = [df]
    for label, cnt in counts.items():
        if cnt < min_count:
            subset = df[df['label'] == label]
            repeat = min(max_multiplier, min_count // max(cnt, 1))
            if repeat > 1:
                for _ in range(repeat - 1):
                    dfs.append(subset)
    result = pd.concat(dfs, ignore_index=True)
    return result.sample(frac=1, random_state=42).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. R-Drop consistency loss
# ---------------------------------------------------------------------------
def rdrop_loss(logits1, logits2, alpha=0.5):
    """KL divergence between two forward-pass output distributions."""
    p1 = F.softmax(logits1, axis=-1)
    p2 = F.softmax(logits2, axis=-1)
    kl_12 = F.kl_div(F.log_softmax(logits1, axis=-1), p2, reduction='batchmean')
    kl_21 = F.kl_div(F.log_softmax(logits2, axis=-1), p1, reduction='batchmean')
    return alpha * (kl_12 + kl_21) / 2.0


# ---------------------------------------------------------------------------
# 7. Training loop
# ---------------------------------------------------------------------------
def run_training(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, f'cpa_{timestamp}')
    setup_logging(save_dir)
    set_seed(args.random_seed)
    device = resolve_device(args.device)

    logging.info(f'device: {device}')
    logging.info(f'model: {args.shortcut_name}')
    logging.info(f'contrastive_alpha: {args.contrastive_alpha}, rdrop_alpha: {args.rdrop_alpha}')

    # ---- 7a. Load data ----
    raw_train_df = load_data_from_directory(args.train_dir)

    # ---- 7b. Build label encoder ----
    label_encoder = LabelEncoder()
    label_encoder.fit(raw_train_df['label'].unique())
    num_classes = len(label_encoder.classes_)
    logging.info(f'label_num: {num_classes}')
    save_label_classes(label_encoder, save_dir)

    # ---- 7c. Competition weights from ORIGINAL distribution ----
    orig_counts = raw_train_df['label'].value_counts()
    counts_max = orig_counts.max()
    counts_min = orig_counts.min()
    class_weights = []
    for cls_name in label_encoder.classes_:
        count_m = orig_counts.get(cls_name, counts_min)
        weight_m = (counts_max - count_m + counts_min * 0.1) / (counts_max + counts_min * 0.1)
        class_weights.append(weight_m)
    class_weights_tensor = paddle.to_tensor(class_weights, dtype='float32')
    logging.info(f'orig distribution: max={counts_max}, min={counts_min}')

    # ---- 7d. Oversampling ----
    raw_train_df = oversample_rare_classes(raw_train_df, min_count=10, max_multiplier=10)
    logging.info(f'after oversampling: {len(raw_train_df)} rows')

    # ---- 7e. Split ----
    after_counts = raw_train_df['label'].value_counts()
    rare_labels = after_counts[after_counts < 2].index
    df_rare = raw_train_df[raw_train_df['label'].isin(rare_labels)]
    df_common = raw_train_df[~raw_train_df['label'].isin(rare_labels)]

    if len(df_common) == 0:
        raise ValueError("data num < 2, can't split")

    train_c, val_c = train_test_split(
        df_common, test_size=args.val_ratio,
        stratify=df_common['label'], random_state=args.random_seed,
    )
    train_df = pd.concat([train_c, df_rare]).sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    val_df = val_c.reset_index(drop=True)
    logging.info(f'split: train={len(train_df)}, val={len(val_df)}')

    # ---- 7f. DataLoaders ----
    tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)
    train_loader = DataLoader(
        RelationDataset(train_df, tokenizer, label_encoder, args.max_length),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=dynamic_collate_fn, num_workers=args.num_workers, return_list=True,
    )
    val_loader = DataLoader(
        RelationDataset(val_df, tokenizer, label_encoder, args.max_length),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=dynamic_collate_fn, num_workers=args.num_workers, return_list=True,
    )

    # ---- 7g. Model ----
    model = CPAModel(args.shortcut_name, num_classes, args.dropout)
    model.init_prototypes_from_names(tokenizer, list(label_encoder.classes_))

    total_steps = max(1, len(train_loader) * args.epoch)
    lr_scheduler = LinearDecayWithWarmup(args.lr, total_steps, warmup=args.warmup_ratio)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler, parameters=model.parameters(),
        weight_decay=args.weight_decay,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
    )
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    cont_loss_fn = nn.CrossEntropyLoss()  # no weight — prototypes handle class balance

    use_amp = args.use_amp and str(device) != 'cpu'
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024) if use_amp else None

    # ---- 7h. Training ----
    best_score = 0.0
    patience_counter = 0

    logging.info('start training...')
    for epoch in range(args.epoch):
        model.train()
        tr_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epoch}')
        for batch in pbar:
            if batch is None:
                continue

            input_ids = paddle.to_tensor(batch['data'], dtype='int64')
            mask = paddle.to_tensor(batch['cls_mask'], dtype='int64')
            label_ids = paddle.to_tensor(batch['label'], dtype='int64')

            if use_amp:
                with paddle.amp.auto_cast(enable=True):
                    # Two forward passes for R-Drop (different dropout)
                    cls1, cont1 = model(input_ids, mask)
                    cls2, cont2 = model(input_ids, mask)

                    ce_loss = (ce_loss_fn(cls1, label_ids) + ce_loss_fn(cls2, label_ids)) / 2.0
                    cont_loss = (cont_loss_fn(cont1, label_ids) + cont_loss_fn(cont2, label_ids)) / 2.0
                    rd_loss = rdrop_loss(cls1, cls2, args.rdrop_alpha)
                    loss = ce_loss + args.contrastive_alpha * cont_loss + rd_loss

                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.minimize(optimizer, scaled)
                optimizer.clear_grad()
            else:
                cls1, cont1 = model(input_ids, mask)
                cls2, cont2 = model(input_ids, mask)

                ce_loss = (ce_loss_fn(cls1, label_ids) + ce_loss_fn(cls2, label_ids)) / 2.0
                cont_loss = (cont_loss_fn(cont1, label_ids) + cont_loss_fn(cont2, label_ids)) / 2.0
                rd_loss = rdrop_loss(cls1, cls2, args.rdrop_alpha)
                loss = ce_loss + args.contrastive_alpha * cont_loss + rd_loss

                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

            lr_scheduler.step()
            tr_loss += float(loss.numpy())
            train_steps += 1
            pbar.set_postfix({'loss': f'{float(loss.numpy()):.4f}'})

        # ---- Validation ----
        model.eval()
        all_preds, all_labels = [], []
        with paddle.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                input_ids = paddle.to_tensor(batch['data'], dtype='int64')
                mask = paddle.to_tensor(batch['cls_mask'], dtype='int64')
                label_ids = paddle.to_tensor(batch['label'], dtype='int64')

                if use_amp:
                    with paddle.amp.auto_cast(enable=True):
                        cls_logits, cont_logits = model(input_ids, mask)
                else:
                    cls_logits, cont_logits = model(input_ids, mask)

                # Combine classifier and contrastive logits
                final_logits = cls_logits + args.contrastive_beta * cont_logits
                preds = paddle.argmax(final_logits, axis=1)
                all_preds.extend(preds.numpy().tolist())
                all_labels.extend(label_ids.numpy().tolist())

        # Competition metric
        m_correct, m_total = defaultdict(int), defaultdict(int)
        for p, l in zip(all_preds, all_labels):
            m_total[l] += 1
            if p == l:
                m_correct[l] += 1

        score_num, score_den = 0.0, 0.0
        for l_idx, w in enumerate(class_weights):
            if m_total[l_idx] > 0:
                m_score = m_correct[l_idx] / m_total[l_idx]
            else:
                m_score = 0.0
            score_num += w * m_score
            score_den += w
        val_score = score_num / score_den if score_den > 0 else 0.0

        avg_loss = tr_loss / max(1, train_steps)
        logging.info(f'Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Val Score: {val_score:.4f}')

        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            paddle.save(model.state_dict(), os.path.join(save_dir, 'best_model.pdparams'))
            try:
                tokenizer.save_pretrained(save_dir)
            except Exception:
                pass
            logging.info(f'best model! (Score: {best_score:.4f})')
        else:
            patience_counter += 1
            logging.info(f'early stop: {patience_counter}/{args.patience}')
            if patience_counter >= args.patience:
                logging.info(f'{args.patience} epochs no improvement, early stop!')
                break

    logging.info(f'training finished. best score: {best_score:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default="./dataset/Train_Set")
    parser.add_argument('--output_dir', type=str, default='./cpa_output')
    parser.add_argument('--shortcut_name', type=str, default='roberta-base')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.1)
    # Dual-encoder contrastive loss weight
    parser.add_argument('--contrastive_alpha', type=float, default=0.3,
                        help='Weight for contrastive prototype loss')
    # R-Drop consistency loss weight
    parser.add_argument('--rdrop_alpha', type=float, default=0.5,
                        help='Weight for R-Drop KL-divergence loss')
    # Inference: how much to trust contrastive logits vs classifier logits
    parser.add_argument('--contrastive_beta', type=float, default=0.5,
                        help='Weight for contrastive logits at inference time')
    args = parser.parse_args()
    run_training(args)
