import os
import logging
import torch
from pathlib import Path
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

logger = logging.getLogger(__name__)

class SpiderDataset(Dataset):
    """Formats Spider JSON data into causal LM training examples (Question -> MongoDB Query)."""

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item.get("question", "")
        mongodb_query = item.get("generated_mongodb_query", "")

        # Handle if stored as a list
        if isinstance(mongodb_query, list) and len(mongodb_query) > 0:
            mongodb_query = mongodb_query[0]

        prompt = f"### Task: Generate MongoDB Query (MQL)\n### Question:\n{question}\n\n### MongoDB Query:\n"

        prompt_enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")
        completion_enc = self.tokenizer(mongodb_query, truncation=True, max_length=self.max_length, return_tensors="pt")

        input_ids = torch.cat([prompt_enc.input_ids[0], completion_enc.input_ids[0]], dim=0)
        attention_mask = torch.cat([prompt_enc.attention_mask[0], completion_enc.attention_mask[0]], dim=0)

        prompt_len = prompt_enc.input_ids.shape[1]
        labels = torch.cat([torch.full((prompt_len,), -100, dtype=torch.long), completion_enc.input_ids[0]], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class ModelLoader:
    def __init__(self, config, device_override=None):
        self.config = config
        self.device = device_override or self._detect_device()
        if self.device == "cuda":
            self.dtype = torch.float16
        elif self.device == "mps":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.tokenizer = None
        self.base_model = None
        self.lora_model = None
        
        logger.info(f"🖥️ Initialized ModelLoader | Device: {self.device} | Dtype: {self.dtype}")

    def _detect_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_tokenizer(self):
        model_name = getattr(self.config, "model_name", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
        logger.info(f"🔄 Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_base_model(self):
        model_name = getattr(self.config, "model_name", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
        logger.info(f"🔄 Loading base model: {model_name}")

        if self.device == "cuda":
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.base_model.to(self.device)

        self.base_model.eval()
        return self.base_model

    def setup_lora(self, trainable_fraction=1.0, r=16, alpha=32, dropout=0.05):
        if self.base_model is None:
            self.load_base_model()
            
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.lora_model = get_peft_model(self.base_model, peft_config)
        
        if trainable_fraction < 1.0:
            self._apply_trainable_fraction(trainable_fraction)
        else:
            logger.info("✅ LoRA configured with 100% trainable parameters.")
            
        self.lora_model.print_trainable_parameters()
        return self.lora_model

    def _apply_trainable_fraction(self, fraction):
        lora_params = [(n, p) for n, p in self.lora_model.named_parameters() if "lora_" in n and p.requires_grad]
        total_lora = len(lora_params)
        num_to_freeze = int(total_lora * (1.0 - fraction))
        
        if num_to_freeze > 0:
            for i, (name, param) in enumerate(lora_params):
                if i < num_to_freeze:
                    param.requires_grad = False
            
            trainable_count = sum(1 for p in self.lora_model.parameters() if p.requires_grad)
            logger.info(f"🔒 Froze {num_to_freeze}/{total_lora} LoRA parameters. Remaining trainable: {trainable_count} ({fraction*100}%).")

    def autodetect_saved_model(self, output_dir):
        output_dir = Path(output_dir)
        if output_dir.exists() and (output_dir / "adapter_config.json").exists():
            logger.info(f"✅ Found saved LoRA adapter at {output_dir}. Loading...")
            if self.base_model is None:
                self.load_base_model()
            self.lora_model = PeftModel.from_pretrained(self.base_model, str(output_dir))
            self.lora_model.to(self.device)
            self.lora_model.eval()
            return self.lora_model
        logger.info(f"ℹ️ No saved adapter found at {output_dir}.")
        return None

    def train_lora(self, train_data, output_dir, epochs=3, batch_size=1, grad_accum=4, lr=2e-4, max_length=512):
        lr = float(lr) # Ensure lr is a float to prevent PyTorch optimizer TypeError
        if self.lora_model is None:
            self.setup_lora()
            
        dataset = SpiderDataset(train_data, self.tokenizer, max_length)
        
        def collate_fn(batch):
            max_len = max(len(b["input_ids"]) for b in batch)
            input_ids, attention_mask, labels = [], [], []
            pad_id = self.tokenizer.pad_token_id
            
            for b in batch:
                pad_len = max_len - len(b["input_ids"])
                input_ids.append(b["input_ids"].tolist() + [pad_id] * pad_len)
                attention_mask.append(b["attention_mask"].tolist() + [0] * pad_len)
                labels.append(b["labels"].tolist() + [-100] * pad_len)
                
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long).to(self.device),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long).to(self.device),
                "labels": torch.tensor(labels, dtype=torch.long).to(self.device)
            }
            
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.lora_model.parameters()), 
            lr=lr
        )
        
        self.lora_model.train()
        
        logger.info(f"🚀 Starting LoRA training | Device: {self.device} | Epochs: {epochs} | Batch: {batch_size} | Grad Accum: {grad_accum} | LR: {lr}")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            optimizer.zero_grad()
            
            for step, batch in enumerate(progress_bar):
                outputs = self.lora_model(**batch)
                loss = outputs.loss / grad_accum
                loss.backward()
                epoch_loss += loss.item() * grad_accum
                
                if (step + 1) % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                progress_bar.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}"})
                
                # Clear MPS cache periodically to prevent memory leaks on Mac
                if self.device == "mps" and (step + 1) % 10 == 0:
                    torch.mps.empty_cache()
                    
            optimizer.step()
            optimizer.zero_grad()
            
            avg_loss = epoch_loss / len(loader)
            logger.info(f"✅ Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.lora_model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        logger.info(f"💾 LoRA adapter saved to {output_dir}")
        
        self.lora_model.eval()
        return self.lora_model

    def load_models(self, lora_path=None):
        """Returns a dictionary of models and the tokenizer."""
        if self.tokenizer is None:
            self.load_tokenizer()
        if self.base_model is None:
            self.load_base_model()

        models = {"base": self.base_model, "lora": None}

        if self.lora_model is not None:
            logger.info("Reusing already-loaded LoRA model in memory.")
            models["lora"] = self.lora_model
        elif lora_path:
            lora_path = Path(lora_path)
            if lora_path.exists() and (lora_path / "adapter_config.json").exists():
                logger.info(f"Loading LoRA model from {lora_path}")
                self.lora_model = PeftModel.from_pretrained(self.base_model, str(lora_path))
                self.lora_model.to(self.device)
                self.lora_model.eval()
                models["lora"] = self.lora_model
            else:
                logger.warning(f"LoRA path {lora_path} does not exist or is invalid. Returning base model as 'lora'.")
                models["lora"] = self.base_model

        return models, self.tokenizer