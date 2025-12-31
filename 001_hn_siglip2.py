import os
import io
import json
import base64
import mmap
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup
from PIL import Image
from tqdm import tqdm
import faiss

# ===========================
# 0. Global Configuration
# ===========================
CONFIG = {
    "model_name": "/ldap_shared/home/s_rlx/hllm/img_text_retrieval/code/local_models/siglip2-base-patch16-224/", 
    "data_root": "./Multimodal_Retrieval",
    "max_len": 32,
    "batch_size":16, 
    "lr": 1e-5,
    "epochs": 5,
    "num_workers":1,
    "device": "cuda:6",
    "seed": 42,
    "hnm_start_epoch": 1,
    "top_k_hard": 10
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(CONFIG['seed'])

# ===========================
# 1. Data Reader & Processing
# ===========================
class TSVImageReader:
    def __init__(self, tsv_path):
        self.tsv_path = tsv_path
        self.offset_dict = {} 
        self._build_index()

    def _build_index(self):
        logger.info(f"Indexing TSV file: {self.tsv_path} ...")
        with open(self.tsv_path, 'r', encoding='utf-8') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line: break
                try:
                    tab_idx = line.find('\t')
                    if tab_idx != -1:
                        img_id = str(line[:tab_idx]).strip()
                        length = len(line.encode('utf-8')) 
                        self.offset_dict[img_id] = (offset, length)
                except Exception:
                    continue
        logger.info(f"Indexed {len(self.offset_dict)} images.")

    def get_image(self, img_id):
        img_id = str(img_id).strip()
        if img_id not in self.offset_dict:
            logger.warning(f"[WARNING] Image ID '{img_id}' NOT FOUND in TSV! Replaced with Black Image.")
            return Image.new('RGB', (224, 224), (0, 0, 0))
        
        offset, length = self.offset_dict[img_id]
        try:
            with open(self.tsv_path, 'rb') as f:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                    mm.seek(offset)
                    line_bytes = mm.read(length)
            
            line_str = line_bytes.decode('utf-8').strip()
            b64_str = line_str.split('\t')[1]
            # [新增] 自动修复 Base64 Padding 问题
            missing_padding = len(b64_str) % 4
            if missing_padding:
                b64_str += '=' * (4 - missing_padding)

            img_data = base64.b64decode(b64_str)
            return Image.open(io.BytesIO(img_data)).convert("RGB")
        except Exception as e:
            logger.warning(f"[WARNING] Decode Error for ID '{img_id}': {e}. Replaced with Black Image.")
            return Image.new('RGB', (224, 224), (0, 0, 0))
    
    def get_all_keys(self):
        return set(self.offset_dict.keys())

class MultiModalDataset(Dataset):
    def __init__(self, queries_path, img_reader, processor, mode='train'):
        self.mode = mode
        self.img_reader = img_reader
        self.processor = processor
        self.samples = []
        self.hard_negatives_cache = {}
        
        valid_img_keys = img_reader.get_all_keys() if img_reader else set()
        
        logger.info(f"Loading queries from {queries_path}")
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    q_id = str(data['query_id']).strip()
                    text = data['query_text']
                    raw_item_ids = [str(x).strip() for x in data.get('item_ids', [])]
                    
                    if mode == 'train':
                        valid_items = [iid for iid in raw_item_ids if iid in valid_img_keys]
                        if not valid_items: continue
                        self.samples.append({
                            "q_id": q_id,
                            "text": text,
                            "item_ids": valid_items 
                        })
                    elif mode == 'val':
                        self.samples.append({"q_id": q_id, "text": text, "item_ids": raw_item_ids})
                    elif mode == 'test':
                        self.samples.append({"q_id": q_id, "text": text})
                except: continue
        
        logger.info(f"Loaded {len(self.samples)} valid samples for {mode}.")

    def update_hard_negatives(self, negatives_dict):
        self.hard_negatives_cache = negatives_dict
        logger.info(f"Dataset updated with {len(negatives_dict)} hard negative lists.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item['text']
        
        text_inputs = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=CONFIG['max_len'], 
            truncation=True, 
            return_tensors="pt",
            return_attention_mask=True 
        )
        if self.mode == 'train':
            # Dynamic sampling
            pos_img_id = random.choice(item['item_ids'])
            pos_image = self.img_reader.get_image(pos_img_id)
            
            neg_image = None
            if item['q_id'] in self.hard_negatives_cache:
                neg_candidates = self.hard_negatives_cache[item['q_id']]
                if neg_candidates:
                    neg_id = random.choice(neg_candidates)
                    neg_image = self.img_reader.get_image(neg_id)
            
            images_to_process = [pos_image]
            if neg_image is not None:
                images_to_process.append(neg_image)
            
            image_inputs = self.processor(images=images_to_process, return_tensors="pt")
            
            return {
                "input_ids": text_inputs.input_ids.squeeze(0),
                "attention_mask": text_inputs.attention_mask.squeeze(0),
                "pixel_values": image_inputs.pixel_values,
                "q_id": item['q_id'],
                "has_hard_neg": (neg_image is not None)
            }
        else:
            return {
                "input_ids": text_inputs.input_ids.squeeze(0),
                "attention_mask": text_inputs.attention_mask.squeeze(0),
                "q_id": item['q_id'],
                "gt_ids": item.get('item_ids', [])
            }

def train_collate_fn(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    
    pixel_values_list = []
    has_neg_flags = []
    
    for x in batch:
        pv = x['pixel_values']
        if pv.shape[0] == 2:
            pixel_values_list.append(pv[0])
            pixel_values_list.append(pv[1])
            has_neg_flags.append(True)
        else:
            pixel_values_list.append(pv[0])
            has_neg_flags.append(False)
            
    pixel_values = torch.stack(pixel_values_list)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'has_neg': all(has_neg_flags)
    }

def val_collate_fn(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    q_ids = [x['q_id'] for x in batch]
    gt_ids = [x['gt_ids'] for x in batch]
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'q_id': q_ids, 'gt_ids': gt_ids}

# ===========================
# 2. Model & Loss
# ===========================
class SiglipLoss(nn.Module):
    def __init__(self, temperature_init=10.0, bias_init=-10.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        self.bias = nn.Parameter(torch.tensor(bias_init))

    def forward(self, image_embeddings, text_embeddings, has_neg=False):
        B = text_embeddings.shape[0]
        img_emb = F.normalize(image_embeddings, p=2, dim=-1)
        txt_emb = F.normalize(text_embeddings, p=2, dim=-1)
        
        logits = torch.matmul(txt_emb, img_emb.t()) * self.temperature.exp() + self.bias
        labels = torch.zeros_like(logits)
        labels[:, :B] = torch.eye(B, device=logits.device)
        labels = labels * 2 - 1
        
        loss = -F.logsigmoid(labels * logits).mean()
        return loss

class RetrievalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(CONFIG['model_name'])
        self.backbone.gradient_checkpointing_enable()
        self.loss_fn = SiglipLoss()

    def forward(self, input_ids, attention_mask, pixel_values, has_neg=False):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        loss = self.loss_fn(outputs.image_embeds, outputs.text_embeds, has_neg=has_neg)
        return loss

    @torch.no_grad()
    def encode_text(self, input_ids, attention_mask):
        return self.backbone.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

    @torch.no_grad()
    def encode_image(self, pixel_values):
        return self.backbone.get_image_features(pixel_values=pixel_values)

# ===========================
# 3. Mining & Evaluation
# ===========================
def build_gallery(model, img_reader, processor, batch_size=256, desc="Gallery"):
    model.eval()
    all_img_ids = list(img_reader.offset_dict.keys())
    
    class GalleryDataset(Dataset):
        def __init__(self, ids): self.ids = ids
        def __len__(self): return len(self.ids)
        def __getitem__(self, idx):
            img_id = self.ids[idx]
            img = img_reader.get_image(img_id)
            inputs = processor(images=img, return_tensors="pt")
            return inputs.pixel_values.squeeze(0), str(img_id)

    loader = DataLoader(GalleryDataset(all_img_ids), batch_size=batch_size, 
                        num_workers=CONFIG['num_workers'], pin_memory=True)
    
    embeddings = []
    ids_list = []
    
    with torch.no_grad():
        for pixel_values, img_ids in tqdm(loader, desc=f"Encoding {desc}"):
            pixel_values = pixel_values.to(CONFIG['device'])
            with autocast():
                feats = model.encode_image(pixel_values)
            feats = F.normalize(feats, p=2, dim=-1)
            embeddings.append(feats.cpu().numpy())
            ids_list.extend(img_ids)
            
    return np.concatenate(embeddings, axis=0), ids_list

def run_mining(model, dataset, img_reader, processor):
    logger.info("Starting Hard Negative Mining...")
    gallery_feats, gallery_ids = build_gallery(model, img_reader, processor, desc="Mining Gallery")
    
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(gallery_feats.shape[1])
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(gallery_feats)
    id_map = {i: str(iid) for i, iid in enumerate(gallery_ids)}
    
    class TextDataset(Dataset):
        def __init__(self, samples): self.samples = samples
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]['text'], self.samples[idx]['q_id'], self.samples[idx]['item_ids']
    
    text_loader = DataLoader(TextDataset(dataset.samples), batch_size=CONFIG['batch_size']*2, 
                             num_workers=CONFIG['num_workers'], collate_fn=lambda x: x)
    
    neg_dict = {}
    with torch.no_grad():
        for batch in tqdm(text_loader, desc="Mining Queries"):
            texts = [x[0] for x in batch]
            q_ids = [x[1] for x in batch]
            gt_ids_list = [set(x[2]) for x in batch]
            
            inputs = processor(text=texts, padding="max_length", max_length=CONFIG['max_len'], truncation=True, return_tensors="pt")
            input_ids = inputs.input_ids.to(CONFIG['device'])
            attention_mask = inputs.attention_mask.to(CONFIG['device'])
            
            with autocast():
                feats = model.encode_text(input_ids, attention_mask)
            feats = F.normalize(feats, p=2, dim=-1).cpu().numpy()
            
            D, I = gpu_index.search(feats, CONFIG['top_k_hard'] + 10)
            
            for i in range(len(q_ids)):
                q_id = q_ids[i]
                gts = gt_ids_list[i]
                found_negs = []
                for idx in I[i]:
                    pred_id = id_map[idx]
                    if pred_id not in gts:
                        found_negs.append(pred_id)
                    if len(found_negs) >= CONFIG['top_k_hard']: break
                neg_dict[q_id] = found_negs
    return neg_dict

def evaluate(model, val_loader, gallery_feats, gallery_ids):
    model.eval()
    logger.info("Evaluating...")
    
    dim = gallery_feats.shape[1]
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(dim)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(gallery_feats)
    
    id_map = {i: str(img_id) for i, img_id in enumerate(gallery_ids)}
    recall_1, recall_5, recall_10, total = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Encoding Queries"):
            input_ids = batch['input_ids'].to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            gt_ids_batch = batch['gt_ids'] 
            
            with autocast():
                text_feats = model.encode_text(input_ids, attention_mask)
            text_feats = F.normalize(text_feats, p=2, dim=-1).cpu().numpy()
            
            D, I = gpu_index.search(text_feats, 10)
            
            for i in range(len(text_feats)):
                retrieved_indices = I[i]
                true_ids = set([str(x).strip() for x in gt_ids_batch[i]])
                
                hits = [1 if id_map[idx] in true_ids else 0 for idx in retrieved_indices]
                if sum(hits[:1]) > 0: recall_1 += 1
                if sum(hits[:5]) > 0: recall_5 += 1
                if sum(hits[:10]) > 0: recall_10 += 1
                total += 1
    
    r1, r5, r10 = recall_1/total, recall_5/total, recall_10/total
    mean_recall = (r1 + r5 + r10) / 3.0
    logger.info(f"R@1: {r1:.4f}, R@5: {r5:.4f}, R@10: {r10:.4f}, Mean: {mean_recall:.4f}")
    return mean_recall

# ===========================
# 4. Main Loop
# ===========================
def main():
    processor = AutoProcessor.from_pretrained(CONFIG['model_name'])
    
    train_img_reader = TSVImageReader(os.path.join(CONFIG['data_root'], "MR_train_imgs.tsv"))
    val_img_reader = TSVImageReader(os.path.join(CONFIG['data_root'], "MR_valid_imgs.tsv"))
    
    train_ds = MultiModalDataset(
        os.path.join(CONFIG['data_root'], "MR_train_queries.jsonl"), 
        train_img_reader, processor, mode='train'
    )
    val_ds = MultiModalDataset(
        os.path.join(CONFIG['data_root'], "MR_valid_queries.jsonl"), 
        None, processor, mode='val'
    )
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
                              num_workers=CONFIG['num_workers'], pin_memory=True, 
                              drop_last=True, collate_fn=train_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size']*2, shuffle=False, 
                            num_workers=CONFIG['num_workers'], collate_fn=val_collate_fn)
    
    model = RetrievalModel().to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    
    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda')
    best_recall = 0.0
    
    for epoch in range(CONFIG['epochs']):
        if epoch == CONFIG['hnm_start_epoch']:
            neg_dict = run_mining(model, train_ds, train_img_reader, processor)
            train_ds.update_hard_negatives(neg_dict)
            
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            pixel_values = batch['pixel_values'].to(CONFIG['device'])
            has_neg = batch['has_neg']
            
            optimizer.zero_grad()
            with autocast():
                loss = model(input_ids, attention_mask, pixel_values, has_neg=has_neg)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "HNM": str(has_neg)})
        
        logger.info(f"Epoch {epoch+1} Avg Loss: {train_loss/len(train_loader):.4f}")
        
        gallery_feats, gallery_ids = build_gallery(model, val_img_reader, processor)
        mean_recall = evaluate(model, val_loader, gallery_feats, gallery_ids)
        
        if mean_recall > best_recall:
            best_recall = mean_recall
            torch.save(model.state_dict(), "best_model.pth")
            logger.info(f"New Best Model Saved (Mean Recall: {mean_recall:.4f})")

    logger.info("Generating Test Predictions...")
    model.load_state_dict(torch.load("best_model.pth"))

    test_img_reader = TSVImageReader(os.path.join(CONFIG['data_root'], "MR_test_imgs.tsv")) 
    test_gallery_feats, test_gallery_ids = build_gallery(model, test_img_reader, processor, desc="Test Gallery")
    
    dim = test_gallery_feats.shape[1]
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(dim)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(test_gallery_feats)
    test_id_map = {i: str(iid) for i, iid in enumerate(test_gallery_ids)}
    
    test_ds = MultiModalDataset(os.path.join(CONFIG['data_root'], "MR_test_queries.jsonl"), None, processor, mode='test')
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size']*2, shuffle=False, num_workers=CONFIG['num_workers'], collate_fn=val_collate_fn)
    
    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            q_ids = batch['q_id']
            
            with autocast():
                text_feats = model.encode_text(input_ids, attention_mask)
            text_feats = F.normalize(text_feats, p=2, dim=-1).cpu().numpy()
            
            D, I = gpu_index.search(text_feats, 10)
            
            for i in range(len(q_ids)):
                pred_item_ids = [int(test_id_map[idx]) for idx in I[i]]
                results.append({"query_id": int(q_ids[i]), "item_ids": pred_item_ids})
    
    with open("prediction.jsonl", 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    logger.info("Done. Check prediction.jsonl")

if __name__ == "__main__":
    main()