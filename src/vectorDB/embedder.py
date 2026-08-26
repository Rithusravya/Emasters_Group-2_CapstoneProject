import logging
import torch
from typing import List, Union
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

class HFEmbedder:
    
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading embedding model: {model_name} on {device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        logger.info(f"✅ Embedding model loaded successfully")
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Encode texts into embeddings with progress bar.
        
        Args:
            texts: Single string or list of strings to encode
            batch_size: Batch size for encoding
            max_length: Maximum token length
            normalize: Whether to L2-normalize embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            Tensor of shape (num_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # Process in batches with progress bar
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding texts", unit="batch")
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token or mean pooling
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu())
        
        # Concatenate all batches
        final_embeddings = torch.cat(all_embeddings, dim=0)
        
        if show_progress:
            logger.info(f"✅ Encoded {len(texts)} texts -> shape {final_embeddings.shape}")
        
        return final_embeddings