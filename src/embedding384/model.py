"""
Model initialization and inference for the embedding pipeline.
"""

import os
import gc
import time
import torch
import multiprocessing
from typing import List, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModel
from .logger import PipelineLogger, InferenceMetrics
from .utils import log_and_reraise

# Configuration settings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256  # Maximum sequence length for tokenization

class OptimizedEmbeddingModel:
    def __init__(self, batch_size: int, quiet: bool = False, logger: PipelineLogger = None):
        self.device = torch.device("mps")
        self.batch_size = batch_size
        self.quiet = quiet
        self.model_name = MODEL_NAME
        self.logger = logger

        if not quiet:
            print(f"\nInitializing model with batch_size={batch_size} on MPS...")

        self._initialize_model()

    def _load_tokenizer(self):
        """Load and configure the tokenizer."""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _load_base_model(self):
        """Load the base model with optimizations."""
        print("Loading base model...")
        base_model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("Moving model to MPS...")
        base_model.to(self.device)
        base_model.eval()
        return base_model

    def _prepare_sample_inputs(self):
        """Prepare sample inputs for model tracing."""
        print("Preparing sample inputs...")
        sample_texts = ["This is a longer initialization text that will ensure adequate buffer sizes"] * self.batch_size
        return self.tokenizer(
            sample_texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        ).to(self.device)

    def _trace_and_optimize(self, base_model, inputs):
        """Trace and optimize the model for inference."""
        print("Tracing model...")
        with torch.inference_mode():
            traced_model = torch.jit.trace(
                base_model,
                (inputs['input_ids'], inputs['attention_mask']),
                strict=False
            )
            print("Optimizing traced model...")
            self.model = torch.jit.optimize_for_inference(traced_model)
            
            # Warmup pass
            _ = self.model(inputs['input_ids'], inputs['attention_mask'])

    def _initialize_buffers(self, inputs):
        """Initialize pre-allocated buffers."""
        print("Initializing buffers...")
        # Pre-allocate larger buffers to avoid resizing
        max_batch = int(self.batch_size * 1.1)  # 10% overhead
        self.input_buffers = {
            'input_ids': torch.zeros((max_batch, MAX_SEQ_LENGTH), dtype=torch.long, device=self.device),
            'attention_mask': torch.zeros((max_batch, MAX_SEQ_LENGTH), dtype=torch.long, device=self.device)
        }
        
        # Initialize output buffer
        with torch.inference_mode():
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            self.output_buffer = torch.zeros(
                (max_batch, embeddings.shape[1]),
                dtype=torch.float16,
                device=self.device
            )

    def _initialize_model(self):
        """Initialize the model with all optimizations."""
        print("\nInitializing model...")
        torch.set_num_threads(multiprocessing.cpu_count())
        gc.collect()
        
        try:
            # Load components
            self._load_tokenizer()
            base_model = self._load_base_model()
            inputs = self._prepare_sample_inputs()
            
            # Trace and optimize
            self._trace_and_optimize(base_model, inputs)
            
            # Initialize buffers
            self._initialize_buffers(inputs)
            
            print("Cleaning up...")
            del base_model, inputs
            gc.collect()
            torch.mps.empty_cache()
            
            print("Running warmup encode...")
            warmup_texts = ["Warm-up sentence"] * (self.batch_size // 4)
            _ = self.encode(warmup_texts)
            
            print("Model initialization complete!")
            
        except Exception as e:
            log_and_reraise(e, "model initialization")

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling with normalization."""
        token_embeddings = model_output['last_hidden_state']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        normalized = sum_embeddings / sum_mask
        return torch.nn.functional.normalize(normalized, p=2, dim=1)

    def encode(self, texts: List[str]) -> Tuple[np.ndarray, float, InferenceMetrics]:
        """Optimized single-process inline quantization."""
        start_time = time.time()
        
        # 1. Tokenize
        tokenize_start = time.time()
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH, 
            return_tensors="pt"
        )
        tokenize_time = time.time() - tokenize_start

        # Move to MPS efficiently
        mps_start = time.time()
        batch_size = len(texts)
        for k, v in inputs.items():
            if k in self.input_buffers:
                # Use pre-allocated buffers
                self.input_buffers[k][:batch_size, :v.size(1)].copy_(v)
                inputs[k] = self.input_buffers[k][:batch_size, :v.size(1)]
        mps_time = time.time() - mps_start

        # 2. Forward pass with inference mode
        with torch.inference_mode():
            forward_start = time.time()
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            forward_time = time.time() - forward_start

            # 3. Pooling
            pool_start = time.time()
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            self.output_buffer[:batch_size].copy_(embeddings)
            float16_embeddings = self.output_buffer[:batch_size]
            pool_time = time.time() - pool_start

        # 4. Quantize efficiently
        quantize_start = time.time()
        emb_cpu = float16_embeddings.detach().cpu().float()
        global_max_abs = torch.max(torch.abs(emb_cpu))
        global_scale = global_max_abs / 127.0
        quantized_cpu = torch.round(emb_cpu / global_scale).clamp(-128, 127).to(torch.int8)
        sim = float(
            torch.nn.functional.cosine_similarity(
                emb_cpu, 
                quantized_cpu.float() * global_scale, 
                dim=1
            ).mean().item()
        )
        quantize_time = time.time() - quantize_start

        # Log memory if needed
        if self.logger:
            self.logger.log_memory()

        total_time = time.time() - start_time
        metrics = InferenceMetrics(
            tokenization_time=tokenize_time,
            model_forward_time=forward_time,
            pooling_time=pool_time,
            quantization_time=quantize_time,
            total_time=total_time,
            batch_size=batch_size,
            texts_per_second=batch_size/total_time if total_time > 0 else 0
        )

        return quantized_cpu.numpy(), sim, metrics 