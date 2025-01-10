"""
MPS-optimized text embedding model using all-MiniLM-L6-v2.
Optimized for Apple Silicon with dynamic batching and buffer pre-allocation.
"""

import torch
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)

class MPSEmbeddingModel:
    def __init__(self, max_length=128, batch_size=128):
        self.device = torch.device("mps")
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self.input_buffers = {}
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model and move it to MPS."""
        logger.info("Initializing model on mps")
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        base_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self._optimize_model(base_model)

    def _optimize_model(self, base_model):
        """Optimize the model for MPS and initialize buffers."""
        self.model = base_model.to(self.device)
        self.model.eval()
        
        # Initialize buffers with a sample input
        sample_texts = ["test"] * self.batch_size
        inputs = self.tokenizer(
            sample_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        self._initialize_buffers(inputs)

    def _initialize_buffers(self, sample_inputs):
        """Initialize input buffers for the model."""
        self.input_buffers = {
            'input_ids': torch.zeros((self.batch_size, self.max_length), dtype=torch.long, device=self.device),
            'attention_mask': torch.zeros((self.batch_size, self.max_length), dtype=torch.long, device=self.device),
            'token_type_ids': torch.zeros((self.batch_size, self.max_length), dtype=torch.long, device=self.device)
        }
        
        # Run a sample forward pass to warm up the model
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in sample_inputs.items()}
            model_output = self.model(**inputs)
            embeddings = self._mean_pooling(model_output, inputs['attention_mask'])
            logger.info(f"Initialized buffers with shape {embeddings.shape}")

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling of token embeddings."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts):
        """Encode a batch of texts to embeddings."""
        batch_size = len(texts)
        
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move inputs to device and use pre-allocated buffers
            for key in inputs:
                if key in self.input_buffers:
                    self.input_buffers[key][:batch_size, :inputs[key].size(1)] = inputs[key].to(self.device)
                    inputs[key] = self.input_buffers[key][:batch_size, :inputs[key].size(1)]
            
            # Forward pass
            model_output = self.model(**inputs)
            embeddings = self._mean_pooling(model_output, inputs['attention_mask'])
            return embeddings.cpu().numpy()

    def get_metrics(self):
        """Return model metrics."""
        return {
            "device": str(self.device),
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "model_name": "all-MiniLM-L6-v2"
        } 