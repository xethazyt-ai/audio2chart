import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EncodecModel, AutoProcessor
from huggingface_hub import hf_hub_download

from inference.model_inference import TransformerDecoderAudioConditioned
from tqdm import tqdm

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
@dataclass
class TransformerConfig:
    vocab_size: int
    pad_token_id: int
    eos_token_id: int
    bos_token_id: int
    d_model: int = 512
    n_heads: int = 8
    num_kv_heads: int = 2
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    audio_drop: float = 0.0
    compression: Optional[int] = None
    rope_base: float = 10000.0
    conditional: bool = False
    use_flash: bool = False
    codebook_size: int = 128,
    grid_ms: int = 20


# ------------------------------------------------------------------
# Inference Model
# ------------------------------------------------------------------
class Charter(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.encoder = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.transformer = TransformerDecoderAudioConditioned(**asdict(config))

        self.encoder.eval()
        self.transformer.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.transformer.parameters():
            p.requires_grad = False

    @classmethod
    def from_pretrained(cls, repo_id: str):
        """Load from a Hugging Face repo id, or from a local directory holding
        config.json and pytorch_model.bin -- which is what export_checkpoint.py writes,
        and the only way to run a model trained here rather than a released one."""
        local = Path(repo_id)
        if (local / "config.json").is_file() and (local / "pytorch_model.bin").is_file():
            with (local / "config.json").open(encoding="utf-8") as handle:
                cfg = TransformerConfig(**json.load(handle))
            model = cls(cfg)
            model.transformer.load_state_dict(
                torch.load(local / "pytorch_model.bin", map_location="cpu")
            )
            return model

        cfg_path = hf_hub_download(repo_id, "config.json")
        with open(cfg_path) as f:
            cfg = TransformerConfig(**json.load(f))
        model = cls(cfg)
        bin_path = hf_hub_download(repo_id, "pytorch_model.bin")
        state = torch.load(bin_path, map_location="cpu")
        model.transformer.load_state_dict(state)
        return model


    def _read_audio(self, audio_path: str, device: torch.device):
        import librosa
        wav, sr = librosa.load(audio_path, sr=24000, mono=True)
        inputs = self.processor(
            raw_audio=wav,
            sampling_rate=24000,
            return_tensors="pt"
        ).to(device)
        return inputs["input_values"], inputs["padding_mask"]


    def generate(
        self,
        audio_path: str,
        temperature: float = -1.0,
        top_k: int = 0,
        class_id: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        allowed_ids: Optional[List[int]] = None,
        guidance: float = 0.0,
    ) -> List[torch.Tensor]:
        """
        Fast batched generation with KV-cache + pre-allocation.
        """
        self.to(device)
        self.eval()

        input_values, padding_mask = self._read_audio(audio_path, device)  # [1,1,T]
        total_samples = input_values.size(-1)
        target_sr = 24000
        chunk_sec = 30
        chunk_samples = chunk_sec * target_sr
        ms_resolution = self.config.grid_ms

        if total_samples < chunk_samples:
            raise ValueError(f"Audio must be >= {chunk_sec}s, got {total_samples/target_sr:.2f}s")


        starts = list(range(0, total_samples, chunk_samples))
        if starts[-1] + chunk_samples > total_samples:
            starts[-1] = max(0, total_samples - chunk_samples)   # force full 30 s

        chunks = []
        masks  = []
        for s in starts:
            e = s + chunk_samples
            chunks.append(input_values[..., s:e])
            masks.append(padding_mask[..., s:e])

        audio_batch = torch.cat(chunks, dim=0).to(device)          # [B,1,720000]
        mask_batch  = torch.cat(masks , dim=0).to(device)          # [B,720000]


        with torch.no_grad():
            enc = self.encoder.encode(audio_batch, mask_batch, bandwidth=3.0)
            codes = enc.audio_codes.squeeze(0)                                  # [B,4,T]
            # sum the 4 codebook embeddings                                     
            audio_emb = sum(self.transformer.codes_embedding[i](codes[:, i]) for i in range(4))
            audio_emb = self.transformer.norm_audio(audio_emb)
            if self.transformer.compression:
                audio_emb = self.transformer.audio_compression(audio_emb)       # (B, T_audio', d_model)


        B = audio_batch.size(0)
        class_ids = (torch.full((B, 1), class_id, dtype=torch.long, device=device)
                    if self.config.conditional else None)


        full_seq_len = int(chunk_sec * 1000 / ms_resolution)   

        ids = torch.full((B, full_seq_len + 1), self.config.bos_token_id,
                        dtype=torch.long, device=device)
        ids[:, 0] = self.config.bos_token_id

        self_cache = [None for _ in range(self.transformer.n_layers)]
        cross_cache = [None for _ in range(self.transformer.n_layers)]
        sample_fn = self._make_sampler(temperature, top_k, device, allowed_ids)

        neg_emb = self._contrast_audio(audio_emb) if guidance > 0 else None
        neg_self = [None for _ in range(self.transformer.n_layers)] if neg_emb is not None else None
        neg_cross = [None for _ in range(self.transformer.n_layers)] if neg_emb is not None else None

        for step in tqdm(range(full_seq_len), desc="Let's rock!"):
            cur_token = ids[:, step:step+1]                     # [B,1]

            logits, self_cache, cross_cache = self.transformer(
                cur_token, audio_emb, attention_mask=None, class_ids=class_ids,
                step=step, use_cache=True, self_cache=self_cache, cross_cache=cross_cache
            )                                                   # logits: [B,1,V]

            if neg_emb is not None:
                # Same history, different music. What survives the subtraction is the
                # part of the prediction this audio is responsible for; scaling it up
                # is the whole point, because measurement says it is real but small.
                neg_logits, neg_self, neg_cross = self.transformer(
                    cur_token, neg_emb, attention_mask=None, class_ids=class_ids,
                    step=step, use_cache=True, self_cache=neg_self, cross_cache=neg_cross
                )
                logits = logits + guidance * (logits - neg_logits)

            next_id = sample_fn(logits[:, -1])                  # [B,1]
            ids[:, step + 1] = next_id.squeeze(-1)

        # drop BOS + extract 'new' tokens for last chunk
        sequences = []
        prev_end = starts[-2] + chunk_samples if len(starts) > 1 else 0
        new_tokens = int((total_samples - prev_end) * 1000 / (target_sr * ms_resolution))

        for b in range(B):
            seq = ids[b, 1:]                                    # remove BOS
            if b == B - 1:                                      # last (possibly overlapping) chunk
                sequences.append(seq[-new_tokens:])
            else:
                sequences.append(seq[:full_seq_len])

        return sequences


    @staticmethod
    def _contrast_audio(audio_emb: torch.Tensor) -> torch.Tensor:
        """The negative branch for guidance: real music, but the wrong music.

        A zeroed or mean embedding is the usual choice, but this model was never
        trained with audio dropout, so it has no notion of "no audio" -- feeding it
        one produces logits from far outside the training distribution, and guidance
        would amplify that garbage rather than the conditioning. Every option here
        stays on the manifold of embeddings the model actually saw.

        With several chunks, the negative for each is a neighbouring chunk. With a
        single chunk there is no neighbour, so the window is rolled halfway: still a
        real passage of this song, just not the one being charted now.
        """
        if audio_emb.size(0) > 1:
            return torch.roll(audio_emb, 1, dims=0)
        return torch.roll(audio_emb, audio_emb.size(1) // 2, dims=1)

    def _style_mask(self, allowed_ids, device: torch.device):
        """A [1, V] boolean of what the style forbids, or None.

        Applied before top-k so the two compose: top-k then picks among what the style
        already permits, rather than the style vetoing whatever top-k happened to choose.
        """
        if allowed_ids is None:
            return None
        blocked = torch.ones(1, self.config.vocab_size, dtype=torch.bool, device=device)
        index = torch.as_tensor(sorted(allowed_ids), dtype=torch.long, device=device)
        if index.numel() == 0:
            raise ValueError("Style permits no tokens; sampling would produce NaN")
        blocked[0, index] = False
        return blocked

    def _make_sampler(self, temperature: float, top_k: int, device: torch.device,
                      allowed_ids=None):
        blocked = self._style_mask(allowed_ids, device)

        if temperature <= 0:                     # greedy
            def sample(logits):
                if blocked is not None:
                    logits = logits.masked_fill(blocked, -float('inf'))
                return logits.argmax(dim=-1, keepdim=True)
            return sample

        def sample(logits):
            if blocked is not None:
                logits = logits.masked_fill(blocked, -float('inf'))

            if temperature != 1.0:
                logits = logits / temperature

            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < topk_vals[..., -1:], -float('inf'))

            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        return sample



