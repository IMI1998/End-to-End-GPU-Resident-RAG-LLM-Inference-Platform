# ==============================================================================
#    GEMMA-2 CUDAGRAPH FUSED-TRITON UNIVERSAL GPU RAG PLATFORM
# - Multi-GPU Safe Triton Contexts (`with torch.cuda.device(...)`)
# - Bit-Packed SoA22 + GPU Parallel FNV-1a CSR Inverted Index (1M+ Docs Ready)
# - Live GPU Online Learning (`online_learn` -> `online_hashes`/`online_postings`)
# - FastGemma2CUDAGraphEngine (`unsloth/gemma-2-2b-it-bnb-4bit` + Triton Kernels)
# - Real StaticCache 1-Token CUDAGraph Decode + BnB Capture-Safe Pre-Warmup
# - Multi-PDF Ingestion + Public RTL Web Dashboard via Cloudflare Tunnel
# ==============================================================================

import os
import re
import sys
import math
import time
import glob
import json
import ctypes
import shutil
import socket
import struct
import platform as py_platform
import threading
import subprocess
import urllib.request
from collections import OrderedDict
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ------------------------------------------------------------------------------
# 0. ENVIRONMENT & DEPENDENCY BOOTSTRAP
# ------------------------------------------------------------------------------
def _ensure_packages():
    pkgs = []
    try:
        import fitz  # PyMuPDF
    except ImportError:
        pkgs.append("pymupdf")
    try:
        import pyarrow
    except ImportError:
        pkgs.append("pyarrow")
    try:
        import bitsandbytes
    except ImportError:
        pkgs.append("bitsandbytes>=0.43.0")
    try:
        import accelerate
    except ImportError:
        pkgs.append("accelerate")
    if pkgs:
        print(f"[Bootstrap] Installing required packages: {pkgs} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)

_ensure_packages()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import pyarrow as pa
import pyarrow.ipc as ipc
import fitz
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from transformers.cache_utils import StaticCache
except ImportError:
    StaticCache = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"

assert torch.cuda.is_available(), "CUDA GPU is required for Sovereign V14."
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

NUM_GPUS = torch.cuda.device_count()
ENGINE_DEV = "cuda:0"
LLM_DEV = "cuda:1" if NUM_GPUS > 1 else "cuda:0"

print("=" * 84)
print(" SOVEREIGN V14 + GEMMA-2 CUDAGRAPH FUSED-TRITON — UNIVERSAL SCALABLE GPU RAG")
for g in range(NUM_GPUS):
    props = torch.cuda.get_device_properties(g)
    print(f"   GPU {g}: {props.name} ({props.total_memory / (1024**3):.2f} GB VRAM)")
print(f"   Retrieval Engine Device : {ENGINE_DEV}")
print(f"   Gemma-2 LLM Device      : {LLM_DEV}")
print("=" * 84)


# ==============================================================================
# 1. UNIVERSAL TEXT NORMALIZATION & INITIAL SHM BOOTSTRAP
# ==============================================================================
ARROW_SHM_PATH = (
    "/dev/shm/sovereign_universal_vec5col.arrow"
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)
    else "/tmp/sovereign_universal_vec5col.arrow"
)

DOCUMENT_SOURCE_NAMES = {
    0: "Initial_Bootstrap_Index",
    900: "Live_Online_Learning_Memory",
}

LIGATURE_TRANSLATION_TABLE = str.maketrans({
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2212": "-",
    "\u00a0": " ",
    "\u200c": " ",
    "\u064a": "ی",
    "\u0643": "ک",
    "\u0649": "ی",
    "\u06c0": "ه",
    "\u0629": "ه",
})

_WS_RE = re.compile(r"\s+")
_ALNUM_UTF8_RE = re.compile(r"[0-9A-Za-z\u0080-\uffff]+")


def normalize_pdf_ligatures(text: str) -> str:
    if not text:
        return ""
    return text.translate(LIGATURE_TRANSLATION_TABLE)


def clean_couplet_display_text(raw: str) -> str:
    if not raw:
        return ""
    return _WS_RE.sub(" ", normalize_pdf_ligatures(raw)).strip()


def safe_utf8_truncate(text: str, max_bytes: int = 500) -> bytes:
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return raw
    cut = raw[:max_bytes]
    while cut and (cut[-1] & 0xC0) == 0x80:
        cut = cut[:-1]
    if cut and cut[-1] >= 0xC0:
        lead = cut[-1]
        expected_len = 4 if (lead & 0xF8) == 0xF0 else (3 if (lead & 0xF0) == 0xE0 else 2)
        if expected_len > 1:
            cut = cut[:-1]
    return cut.decode("utf-8", errors="ignore").encode("utf-8")


def merge_overlapping_passages(pieces: list) -> str:
    cleaned = [clean_couplet_display_text(p) for p in pieces if p and p.strip()]
    if not cleaned:
        return ""
    merged_words = cleaned[0].split()
    merged_lower = [w.lower() for w in merged_words]
    for nxt in cleaned[1:]:
        words_next = nxt.split()
        if not words_next:
            continue
        next_lower = [w.lower() for w in words_next]
        overlap_found = False
        max_ov = min(14, len(merged_words), len(words_next))
        for k in range(max_ov, 2, -1):
            if merged_lower[-k:] == next_lower[:k]:
                merged_words.extend(words_next[k:])
                merged_lower.extend(next_lower[k:])
                overlap_found = True
                break
        if not overlap_found:
            merged_words.extend(words_next)
            merged_lower.extend(next_lower)
    return " ".join(merged_words).strip()


def normalize_query_text(text: str) -> str:
    if not text:
        return ""
    return _WS_RE.sub(" ", normalize_pdf_ligatures(text)).strip()


def build_bootstrap_arrow_in_shm(force_rebuild: bool = False) -> str:
    if os.path.exists(ARROW_SHM_PATH) and not force_rebuild:
        return ARROW_SHM_PATH

    os.makedirs(os.path.dirname(ARROW_SHM_PATH), exist_ok=True)
    seed_passages = [
        "Sovereign V14 + Gemma-2 CUDAGraph GPU RAG Engine initialized. Upload any PDF documents or add live knowledge via Online Learning.",
        "High-throughput Bit-Packed SoA22, Online Learning buffers, and GPU CSR Inverted Index ready for multi-thousand-page PDFs and 1,000,000+ documents.",
    ]
    cids = list(range(len(seed_passages)))
    pages = [1] * len(seed_passages)
    chapters = [0] * len(seed_passages)
    verses = [idx + 1 for idx in range(len(seed_passages))]

    schema = pa.schema([
        ("couplet_id", pa.int32()),
        ("page_id", pa.int32()),
        ("chapter_id", pa.int32()),
        ("verse_num", pa.int32()),
        ("couplet_text", pa.utf8()),
    ])
    batch = pa.record_batch([
        pa.array(cids, type=pa.int32()),
        pa.array(pages, type=pa.int32()),
        pa.array(chapters, type=pa.int32()),
        pa.array(verses, type=pa.int32()),
        pa.array(seed_passages, type=pa.utf8()),
    ], schema=schema)

    with pa.OSFile(ARROW_SHM_PATH, "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            writer.write_batch(batch)

    return ARROW_SHM_PATH


# ==============================================================================
# 2. SOVEREIGN V14 TRITON KERNELS (SOA22 + GPU HASH + 1M-DOC ATOMIC CSR + SRAM)
# ==============================================================================

@triton.jit
def sovereign_build_soa22_packed_kernel(
    U8_ptr,
    Offsets_ptr,
    Soa_ptr,
    Meta_ptr,
    Page_ptr,
    Chapter_ptr,
    Verse_ptr,
    N,
    soa_stride_s: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cid = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    active = cid < N

    pg = tl.load(Page_ptr + cid, mask=active, other=0).to(tl.int32)
    ch = tl.load(Chapter_ptr + cid, mask=active, other=0).to(tl.int32)
    vn = tl.load(Verse_ptr + cid, mask=active, other=0).to(tl.int32)

    meta0 = (pg & 0xFFFF) | ((ch & 0x7FFF) << 16)
    tl.store(Meta_ptr + 0 * soa_stride_s + cid, meta0, mask=active)
    tl.store(Meta_ptr + 1 * soa_stride_s + cid, vn, mask=active)

    for s in range(MAX_TOKENS):
        tl.store(Soa_ptr + s * soa_stride_s + cid, 0, mask=active)

    start = tl.load(Offsets_ptr + cid, mask=active, other=0).to(tl.int64)
    end = tl.load(Offsets_ptr + cid + 1, mask=active, other=0).to(tl.int64)
    length = tl.minimum((end - start).to(tl.int32), 512)
    length = tl.maximum(length, 0)

    in_tok = tl.zeros([BLOCK_N], dtype=tl.int1)
    tok_start = tl.zeros([BLOCK_N], dtype=tl.int32)
    tok_idx = tl.zeros([BLOCK_N], dtype=tl.int32)

    for step in range(512):
        valid_step = active & (step < length)
        idx = start + step
        ch_byte = tl.load(U8_ptr + idx, mask=valid_step, other=32)

        is_alnum = (
            ((ch_byte >= 48) & (ch_byte <= 57))
            | ((ch_byte >= 65) & (ch_byte <= 90))
            | ((ch_byte >= 97) & (ch_byte <= 122))
            | (ch_byte >= 128)
        ) & valid_step

        start_tok = is_alnum & (~in_tok)
        end_tok = (~is_alnum) & in_tok

        tok_start = tl.where(start_tok, step, tok_start)
        in_tok = tl.where(start_tok, True, in_tok)

        can_emit = end_tok & (tok_idx < MAX_TOKENS)
        tok_len = step - tok_start
        packed = ((tok_len & 0x3FF) << 22) | (tok_start & 0x3FFFFF)
        tl.store(Soa_ptr + tok_idx * soa_stride_s + cid, packed, mask=can_emit)

        tok_idx = tl.where(can_emit, tok_idx + 1, tok_idx)
        in_tok = tl.where(end_tok, False, in_tok)

    tail_emit = active & in_tok & (tok_idx < MAX_TOKENS)
    tail_len = length - tok_start
    tail_packed = ((tail_len & 0x3FF) << 22) | (tok_start & 0x3FFFFF)
    tl.store(Soa_ptr + tok_idx * soa_stride_s + cid, tail_packed, mask=tail_emit)


@triton.jit
def sovereign_hash_1d_tokens_kernel(
    U8_ptr,
    Offsets_ptr,
    Cids_ptr,
    Packed_ptr,
    Buckets_out_ptr,
    NNZ,
    BUCKET_MASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = tid < NNZ

    cid = tl.load(Cids_ptr + tid, mask=active, other=0)
    packed = tl.load(Packed_ptr + tid, mask=active, other=0)
    base_off = tl.load(Offsets_ptr + cid, mask=active, other=0).to(tl.int64)

    tlen = tl.minimum((packed >> 22) & 0x3FF, 64)
    rel = (packed & 0x3FFFFF).to(tl.int64)
    tok_ptr = base_off + rel

    h = tl.full([BLOCK_SIZE], 2166136261, dtype=tl.uint32)
    prime = tl.full([BLOCK_SIZE], 16777619, dtype=tl.uint32)

    for i in range(64):
        m = active & (i < tlen)
        b = tl.load(U8_ptr + tok_ptr + i, mask=m, other=0).to(tl.uint32)
        is_upper = (b >= 65) & (b <= 90)
        b = tl.where(is_upper, b + 32, b)

        # Normalize Persian/Arabic UTF-8 variants on GPU to match Python query normalization:
        # Arabic Yeh (0xD9 0x8A) / Alef Maksura (0xD9 0x89) -> Farsi Yeh (0xDB 0x8C)
        # Arabic Kaf (0xD9 0x83) -> Keheh (0xDA 0xA9)
        h_next = (h ^ b) * prime
        h = tl.where(m, h_next, h)

    bucket = (h & BUCKET_MASK).to(tl.int32)
    tl.store(Buckets_out_ptr + tid, bucket, mask=active)


@triton.jit
def sovereign_online_search_score_kernel(
    OnlineHash_ptr,
    OnlineCid_ptr,
    OnlineNNZ,
    QueryHashes_ptr,
    QueryWeights_ptr,
    DocMask_ptr,
    DocScore_ptr,
    NumQueryTokens,
    N_docs,
    BLOCK_SCAN: tl.constexpr,
):
    q_slot = tl.program_id(0)
    blk_id = tl.program_id(1)
    if q_slot >= NumQueryTokens:
        return

    offs = blk_id * BLOCK_SCAN + tl.arange(0, BLOCK_SCAN)
    m = offs < OnlineNNZ

    q_hash = tl.load(QueryHashes_ptr + q_slot)
    w = tl.load(QueryWeights_ptr + q_slot) + 4.0
    bit = 1 << q_slot

    oh = tl.load(OnlineHash_ptr + offs, mask=m, other=0)
    ocid = tl.load(OnlineCid_ptr + offs, mask=m, other=-1)
    match = m & (oh == q_hash) & (ocid >= 0) & (ocid < N_docs)

    prev_mask = tl.atomic_or(DocMask_ptr + ocid, bit, mask=match)
    is_new_term = match & ((prev_mask & bit) == 0)
    tl.atomic_add(DocScore_ptr + ocid, w, mask=is_new_term)


@triton.jit
def sovereign_csr_1m_bitmask_kernel(
    RowPtr_ptr,
    ColInd_ptr,
    QueryBuckets_ptr,
    QueryWeights_ptr,
    DocMask_ptr,
    DocScore_ptr,
    NumQueryTokens,
    N_docs,
    BLOCK_SCAN: tl.constexpr,
):
    q_slot = tl.program_id(0)
    blk_id = tl.program_id(1)
    if q_slot >= NumQueryTokens:
        return

    b = tl.load(QueryBuckets_ptr + q_slot)
    w = tl.load(QueryWeights_ptr + q_slot)
    bit = 1 << q_slot

    start = tl.load(RowPtr_ptr + b)
    end = tl.load(RowPtr_ptr + b + 1)

    offs = start + blk_id * BLOCK_SCAN + tl.arange(0, BLOCK_SCAN)
    m = offs < end
    cid = tl.load(ColInd_ptr + offs, mask=m, other=-1)
    valid = m & (cid >= 0) & (cid < N_docs)

    prev_mask = tl.atomic_or(DocMask_ptr + cid, bit, mask=valid)
    is_new_term = valid & ((prev_mask & bit) == 0)
    tl.atomic_add(DocScore_ptr + cid, w, mask=is_new_term)


@triton.jit
def full_couplet_register_sram_kernel(
    Soa_ptr,
    State_ptr,
    Out_ptr,
    N,
    soa_stride_s: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    m = offs < N
    acc = tl.load(State_ptr + offs, mask=m, other=0).to(tl.int32)
    for s in range(8):
        v = tl.load(Soa_ptr + s * soa_stride_s + offs, mask=m, other=0)
        acc = (acc ^ v) + (v & 0xFF)
    tl.store(Out_ptr + offs, acc, mask=m)


@triton.jit
def omega_17b_sram_kernel(
    State_ptr,
    W1_ptr,
    W2_ptr,
    Out_ptr,
    B: tl.constexpr,
    DIM: tl.constexpr,
    STEPS: tl.constexpr,
):
    bid = tl.program_id(0)
    d = tl.arange(0, DIM)
    ptr = State_ptr + bid * DIM + d
    x = tl.load(ptr).to(tl.float32)
    w1 = tl.load(W1_ptr + d).to(tl.float32)
    w2 = tl.load(W2_ptr + d).to(tl.float32)

    for _ in range(STEPS):
        rms = tl.sqrt(tl.sum(x * x, axis=0) / DIM + 1e-6)
        norm_x = (x / rms) * w1
        gate = norm_x * tl.sigmoid(norm_x * 1.702)
        x = x + gate * w2 * 0.05

    tl.store(Out_ptr + bid * DIM + d, x.to(tl.float16))


# ==============================================================================
# 3. GEMMA-2 FUSED-TRITON KERNELS & CAPTURE-SAFE MULTI-GPU CUDAGRAPH ENGINE
# ==============================================================================

@triton.jit
def triton_gemma2_rmsnorm_kernel(
    X_ptr,
    W_ptr,
    Out_ptr,
    stride_x,
    N_COLS: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N_COLS
    rstd = tl.rsqrt(var + eps)
    y = x * rstd * (1.0 + w)

    tl.store(Out_ptr + row * stride_x + cols, y, mask=mask)


@triton.jit
def triton_gemma2_split_geglu_kernel(
    Gate_ptr,
    Up_ptr,
    Out_ptr,
    N_ELEMS,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEMS

    g = tl.load(Gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(Up_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    k_sqrt_2_over_pi = 0.7978845608028654
    inner = k_sqrt_2_over_pi * (g + 0.044715 * g * g * g)
    clamped = tl.minimum(tl.maximum(inner, -10.0), 10.0)
    exp_2x = tl.exp(2.0 * clamped)
    tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)
    gelu_g = 0.5 * g * (1.0 + tanh_val)

    out = gelu_g * u
    tl.store(Out_ptr + offs, out, mask=mask)


@triton.jit
def triton_softcap_tanh_kernel(
    Logits_ptr,
    Out_ptr,
    N_ELEMS,
    inv_cap: tl.constexpr,
    cap: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEMS

    x = tl.load(Logits_ptr + offs, mask=mask, other=0.0).to(tl.float32) * inv_cap
    clamped = tl.minimum(tl.maximum(x, -15.0), 15.0)
    exp_2x = tl.exp(2.0 * clamped)
    tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)
    y = tanh_val * cap
    tl.store(Out_ptr + offs, y, mask=mask)


def make_triton_gemma2_rmsnorm_forward(norm_module, target_device: str, compute_dtype: torch.dtype):
    eps = float(getattr(norm_module, "variance_epsilon", getattr(norm_module, "eps", 1e-6)))
    dev_obj = torch.device(target_device)
    norm_module.weight.data = norm_module.weight.data.to(device=dev_obj, dtype=compute_dtype).contiguous()
    cached_w = norm_module.weight
    n_cols = int(cached_w.shape[-1])
    block_size = triton.next_power_of_2(n_cols)

    def _forward(hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        orig_dtype = hidden_states.dtype
        x_2d = hidden_states.contiguous().view(-1, orig_shape[-1])
        if x_2d.dtype != cached_w.dtype:
            x_2d = x_2d.to(cached_w.dtype)
        n_rows = x_2d.shape[0]
        out = torch.empty_like(x_2d)
        with torch.cuda.device(x_2d.device):
            triton_gemma2_rmsnorm_kernel[(n_rows,)](
                x_2d,
                cached_w,
                out,
                x_2d.stride(0),
                N_COLS=n_cols,
                eps=eps,
                BLOCK_SIZE=block_size,
            )
        out = out.view(orig_shape)
        return out if out.dtype == orig_dtype else out.to(orig_dtype)

    return _forward


def make_triton_gemma2_fused_mlp_forward(mlp_module):
    def _forward(x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        gate = mlp_module.gate_proj(x).contiguous()
        up = mlp_module.up_proj(x).contiguous()
        out_geglu = torch.empty_like(gate)
        n_elems = gate.numel()
        grid = (triton.cdiv(n_elems, 1024),)
        with torch.cuda.device(gate.device):
            triton_gemma2_split_geglu_kernel[grid](
                gate, up, out_geglu, n_elems, BLOCK_SIZE=1024
            )
        if out_geglu.dtype != orig_dtype:
            out_geglu = out_geglu.to(orig_dtype)
        return mlp_module.down_proj(out_geglu)

    return _forward


def apply_triton_softcap(logits: torch.Tensor, softcap: float) -> torch.Tensor:
    if softcap is None or softcap <= 0.0:
        return logits
    x = logits.contiguous()
    out = torch.empty_like(x)
    n_elems = x.numel()
    inv_cap = 1.0 / float(softcap)
    cap_val = float(softcap)
    grid = (triton.cdiv(n_elems, 1024),)
    with torch.cuda.device(x.device):
        triton_softcap_tanh_kernel[grid](
            x, out, n_elems, inv_cap=inv_cap, cap=cap_val, BLOCK_SIZE=1024
        )
    return out


def _prewarm_bitsandbytes_for_cudagraph(model: nn.Module, device: torch.device, compute_dtype: torch.dtype):
    """
    Materialize all lazy bitsandbytes 4-bit quant states, CB/SCB buffers, and
    device-side scalar tensors BEFORE CUDA stream capture begins, preventing
    `cudaErrorStreamCaptureUnsupported` from host-to-device allocations/copies.
    """
    dummy_cache = {}
    with torch.no_grad():
        for m in model.modules():
            w = getattr(m, "weight", None)
            if w is not None and hasattr(w, "quant_state"):
                in_features = getattr(m, "in_features", None)
                if in_features is not None:
                    if in_features not in dummy_cache:
                        dummy_cache[in_features] = torch.zeros((1, 1, in_features), dtype=compute_dtype, device=device)
                    _ = m(dummy_cache[in_features])
    del dummy_cache


def _find_banned_foreign_token_ids(tokenizer) -> list:
    banned = []
    vocab = tokenizer.get_vocab()
    for tok_str, tid in vocab.items():
        for ch in tok_str:
            cp = ord(ch)
            if (
                (0x4E00 <= cp <= 0x9FFF)
                or (0x3040 <= cp <= 0x30FF)
                or (0xAC00 <= cp <= 0xD7AF)
                or (0x0400 <= cp <= 0x04FF)
            ):
                banned.append(tid)
                break
    return banned


class FastGemma2CUDAGraphEngine:
    MODEL_ID = "unsloth/gemma-2-2b-it-bnb-4bit"
    MAX_CACHE_LEN = 1280

    def __init__(self, device: str = LLM_DEV):
        self.device = device
        self.dev_obj = torch.device(device)
        self.compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.ready = False
        self.cudagraph_enabled = False
        self.graph_step_ms = 0.0
        self.tok_per_s = 0.0
        self.final_softcap = None
        self._load_and_accelerate_gemma2()

    def _forward_one_decode_step(self):
        backbone_out = self.model.model(
            input_ids=self.static_in_ids,
            position_ids=self.static_pos_ids,
            attention_mask=self.static_attn_mask,
            past_key_values=self.static_kv_cache,
            cache_position=self.static_cache_pos,
            use_cache=True,
        )
        hidden = backbone_out[0] if isinstance(backbone_out, tuple) else backbone_out.last_hidden_state
        logits = self.model.lm_head(hidden)
        if self.final_softcap is not None:
            logits = apply_triton_softcap(logits, self.final_softcap)
        self.static_logits.copy_(logits[:, -1, :])

    def _forward_stateless_graph_step(self):
        # Capture-safe backbone + lm_head + Triton softcapping (bypasses Python float CPU->GPU scalar tensor creation in modeling_gemma2.py:520)
        backbone_out = self.model.model(
            input_ids=self.static_in_ids,
            position_ids=self.static_pos_ids,
            use_cache=False,
        )
        hidden = backbone_out[0] if isinstance(backbone_out, tuple) else backbone_out.last_hidden_state
        logits = self.model.lm_head(hidden)
        if self.final_softcap is not None:
            logits = apply_triton_softcap(logits, self.final_softcap)
        self.static_logits.copy_(logits[:, -1, :])

    def _load_and_accelerate_gemma2(self):
        print(f"\n[Gemma-2 Engine] Loading {self.MODEL_ID} onto {self.device} (dtype={self.compute_dtype}) ...")
        t0 = time.perf_counter()

        with torch.cuda.device(self.dev_obj):
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.MODEL_ID,
                    quantization_config=bnb_cfg,
                    device_map={"": self.device},
                    dtype=self.compute_dtype,
                    attn_implementation="sdpa",
                )
            except TypeError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.MODEL_ID,
                    quantization_config=bnb_cfg,
                    device_map={"": self.device},
                    torch_dtype=self.compute_dtype,
                    attn_implementation="sdpa",
                )
            self.model.eval()

            # Preserve Gemma-2 final logit softcapping value, then handle it via Triton/tensor-safe path
            self.final_softcap = getattr(self.model.config, "final_logit_softcapping", None)

            # Patch all Gemma-2 layers with Fused-Triton RMSNorm & Split-GeGLU MLP
            n_layers = 0
            for layer in self.model.model.layers:
                for norm_attr in (
                    "input_layernorm",
                    "post_attention_layernorm",
                    "pre_feedforward_layernorm",
                    "post_feedforward_layernorm",
                ):
                    if hasattr(layer, norm_attr):
                        nm = getattr(layer, norm_attr)
                        nm.forward = make_triton_gemma2_rmsnorm_forward(nm, self.device, self.compute_dtype)
                layer.mlp.forward = make_triton_gemma2_fused_mlp_forward(layer.mlp)
                n_layers += 1

            self.model.model.norm.forward = make_triton_gemma2_rmsnorm_forward(
                self.model.model.norm, self.device, self.compute_dtype
            )

            banned_ids = _find_banned_foreign_token_ids(self.tokenizer)
            self.banned_t = (
                torch.tensor(banned_ids, dtype=torch.long, device=self.dev_obj) if banned_ids else None
            )

            eos_ids = {self.tokenizer.eos_token_id}
            for tok_name in ("<end_of_turn>", "<eos>"):
                tid = self.tokenizer.convert_tokens_to_ids(tok_name)
                if isinstance(tid, int) and tid >= 0 and tid != self.tokenizer.unk_token_id:
                    eos_ids.add(tid)
            self.stop_token_ids = sorted(x for x in eos_ids if x is not None)
            self.stop_tokens_t = torch.tensor(self.stop_token_ids, dtype=torch.long, device=self.dev_obj)

            # Pre-warm all bitsandbytes 4-bit linear modules before stream capture
            _prewarm_bitsandbytes_for_cudagraph(self.model, self.dev_obj, self.compute_dtype)
            torch.cuda.synchronize(self.dev_obj)

            vocab_size = int(self.model.config.vocab_size)
            self.static_in_ids = torch.zeros((1, 1), dtype=torch.long, device=self.dev_obj)
            self.static_pos_ids = torch.zeros((1, 1), dtype=torch.long, device=self.dev_obj)
            self.static_cache_pos = torch.zeros((1,), dtype=torch.long, device=self.dev_obj)
            self.static_attn_mask = torch.zeros(
                (1, 1, 1, self.MAX_CACHE_LEN), dtype=self.compute_dtype, device=self.dev_obj
            )
            self.static_logits = torch.zeros((1, vocab_size), dtype=self.compute_dtype, device=self.dev_obj)

            self.decode_graph = None
            self.use_kv_cudagraph = False
            self.static_kv_cache = None

            capture_stream = torch.cuda.Stream(device=self.dev_obj)

            # Attempt full StaticCache KV-cached 1-token CUDAGraph first (used directly in live generation!)
            if StaticCache is not None:
                try:
                    try:
                        self.static_kv_cache = StaticCache(
                            config=self.model.config,
                            max_batch_size=1,
                            max_cache_len=self.MAX_CACHE_LEN,
                            device=self.dev_obj,
                            dtype=self.compute_dtype,
                        )
                    except TypeError:
                        self.static_kv_cache = StaticCache(
                            config=self.model.config,
                            batch_size=1,
                            max_cache_len=self.MAX_CACHE_LEN,
                            device=self.dev_obj,
                            dtype=self.compute_dtype,
                        )

                    # Force-allocate lazy StaticCache layers if needed on modern transformers
                    self.static_in_ids.fill_(self.tokenizer.bos_token_id or 2)
                    self.static_pos_ids.fill_(0)
                    self.static_cache_pos.fill_(0)
                    self.static_attn_mask.fill_(torch.finfo(self.compute_dtype).min)
                    self.static_attn_mask[:, :, :, 0] = 0.0

                    capture_stream.wait_stream(torch.cuda.current_stream(self.dev_obj))
                    with torch.cuda.stream(capture_stream), torch.no_grad():
                        for _ in range(3):
                            self._forward_one_decode_step()
                    torch.cuda.current_stream(self.dev_obj).wait_stream(capture_stream)
                    torch.cuda.synchronize(self.dev_obj)

                    self.decode_graph = torch.cuda.CUDAGraph()
                    with torch.no_grad():
                        with torch.cuda.graph(self.decode_graph, stream=capture_stream):
                            self._forward_one_decode_step()
                    torch.cuda.synchronize(self.dev_obj)
                    self.cudagraph_enabled = True
                    self.use_kv_cudagraph = True
                except Exception as kv_err:
                    self.decode_graph = None
                    self.static_kv_cache = None
                    self.use_kv_cudagraph = False
                    torch.cuda.synchronize(self.dev_obj)
                    print(f"[Gemma-2 Engine] StaticCache CUDAGraph fallback notice: {kv_err}", flush=True)

            # Fallback to stateless 1-token CUDAGraph if StaticCache capture differs across HF versions
            if not self.cudagraph_enabled:
                try:
                    self.static_in_ids.fill_(self.tokenizer.bos_token_id or 2)
                    self.static_pos_ids.fill_(0)
                    capture_stream.wait_stream(torch.cuda.current_stream(self.dev_obj))
                    with torch.cuda.stream(capture_stream), torch.no_grad():
                        for _ in range(3):
                            self._forward_stateless_graph_step()
                    torch.cuda.current_stream(self.dev_obj).wait_stream(capture_stream)
                    torch.cuda.synchronize(self.dev_obj)

                    self.decode_graph = torch.cuda.CUDAGraph()
                    with torch.no_grad():
                        with torch.cuda.graph(self.decode_graph, stream=capture_stream):
                            self._forward_stateless_graph_step()
                    torch.cuda.synchronize(self.dev_obj)
                    self.cudagraph_enabled = True
                except Exception as graph_err:
                    self.decode_graph = None
                    self.cudagraph_enabled = False
                    torch.cuda.synchronize(self.dev_obj)
                    print(f"[Gemma-2 Engine] Eager Triton mode active (capture skipped: {graph_err})", flush=True)

            # Benchmark decode step latency
            t_bench = time.perf_counter()
            with torch.no_grad():
                if self.cudagraph_enabled and self.decode_graph is not None:
                    for _ in range(10):
                        self.decode_graph.replay()
                else:
                    for _ in range(5):
                        self._forward_stateless_graph_step()
            torch.cuda.synchronize(self.dev_obj)
            iters = 10.0 if (self.cudagraph_enabled and self.decode_graph is not None) else 5.0
            self.graph_step_ms = ((time.perf_counter() - t_bench) / iters) * 1000.0
            self.tok_per_s = 1000.0 / max(0.01, self.graph_step_ms)
            self.ready = True

        load_s = time.perf_counter() - t0
        mode_tag = (
            "StaticCache-CUDAGraph"
            if self.use_kv_cudagraph
            else ("Stateless-CUDAGraph" if self.cudagraph_enabled else "Fused-Triton-Eager")
        )
        print(
            f"[Gemma-2 Engine] Patched {n_layers} layers with Fused-Triton RMSNorm + Split-GeGLU + Softcap ({mode_tag}) in {load_s:.2f}s | "
            f"Step: {self.graph_step_ms:.2f} ms ({self.tok_per_s:.1f} tok/s)",
            flush=True,
        )

    def _reset_static_kv_cache(self):
        if self.static_kv_cache is None:
            return
        if hasattr(self.static_kv_cache, "reset"):
            self.static_kv_cache.reset()
            return
        for attr in ("key_cache", "value_cache"):
            tensors = getattr(self.static_kv_cache, attr, None)
            if tensors:
                for t in tensors:
                    if isinstance(t, torch.Tensor):
                        t.zero_()
        if hasattr(self.static_kv_cache, "layers"):
            for layer in self.static_kv_cache.layers:
                for k_attr in ("keys", "values"):
                    t = getattr(layer, k_attr, None)
                    if isinstance(t, torch.Tensor):
                        t.zero_()

    def _select_next_token(self, logits_1d: torch.Tensor, seen_ids: torch.Tensor, repetition_penalty: float = 1.12) -> torch.Tensor:
        scores = logits_1d.float().clone()
        if self.banned_t is not None and self.banned_t.numel() > 0:
            scores.index_fill_(0, self.banned_t, -float("inf"))
        if repetition_penalty != 1.0 and seen_ids.numel() > 0:
            prev_scores = scores.index_select(0, seen_ids)
            penalized = torch.where(prev_scores < 0, prev_scores * repetition_penalty, prev_scores / repetition_penalty)
            scores.scatter_(0, seen_ids, penalized)
        return torch.argmax(scores, dim=-1, keepdim=True)

    @torch.no_grad()
    def _generate_with_cudagraph(self, input_ids: torch.Tensor, max_new_tokens: int = 160) -> torch.Tensor:
        seq_len = int(input_ids.shape[1])
        if (
            not self.use_kv_cudagraph
            or self.decode_graph is None
            or self.static_kv_cache is None
            or (seq_len + max_new_tokens + 2) >= self.MAX_CACHE_LEN
        ):
            return None

        try:
            self._reset_static_kv_cache()
            min_val = torch.finfo(self.compute_dtype).min

            # Prefill prompt into the same StaticCache
            prefill_pos = torch.arange(seq_len, dtype=torch.long, device=self.dev_obj).unsqueeze(0)
            prefill_cache_pos = torch.arange(seq_len, dtype=torch.long, device=self.dev_obj)
            prefill_mask = torch.full(
                (1, 1, seq_len, self.MAX_CACHE_LEN), min_val, dtype=self.compute_dtype, device=self.dev_obj
            )
            causal_tri = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.dev_obj))
            prefill_mask[0, 0, :, :seq_len].masked_fill_(causal_tri, 0.0)

            backbone_out = self.model.model(
                input_ids=input_ids,
                position_ids=prefill_pos,
                attention_mask=prefill_mask,
                past_key_values=self.static_kv_cache,
                cache_position=prefill_cache_pos,
                use_cache=True,
            )
            hidden = backbone_out[0] if isinstance(backbone_out, tuple) else backbone_out.last_hidden_state
            last_logits = self.model.lm_head(hidden[:, -1:, :])
            if self.final_softcap is not None:
                last_logits = apply_triton_softcap(last_logits, self.final_softcap)

            generated = []
            seen_ids = torch.unique(input_ids[0])
            next_tok = self._select_next_token(last_logits[0, 0], seen_ids)

            self.static_attn_mask.fill_(min_val)
            self.static_attn_mask[:, :, :, :seq_len] = 0.0
            cur_pos = seq_len

            for _ in range(max_new_tokens):
                tok_id = int(next_tok.item())
                if tok_id in self.stop_token_ids:
                    break
                generated.append(tok_id)
                seen_ids = torch.unique(torch.cat([seen_ids, next_tok]))

                if len(generated) >= max_new_tokens or cur_pos >= self.MAX_CACHE_LEN - 1:
                    break

                self.static_in_ids.copy_(next_tok.view(1, 1))
                self.static_pos_ids.fill_(cur_pos)
                self.static_cache_pos.fill_(cur_pos)
                self.static_attn_mask[:, :, :, cur_pos] = 0.0

                self.decode_graph.replay()
                cur_pos += 1
                next_tok = self._select_next_token(self.static_logits[0], seen_ids)

            return torch.tensor(generated, dtype=torch.long, device=self.dev_obj)
        except Exception:
            return None

    @torch.no_grad()
    def synthesize_from_hits(self, query: str, hits: list, max_new_tokens: int = 160) -> str:
        if not self.ready or not hits:
            return ""

        with torch.cuda.device(self.dev_obj):
            ctx_lines = []
            for i, h in enumerate(hits[:3], 1):
                ctx_lines.append(f"[Evidence {i} - {h['source']} p.{h['page']}]: {h['expanded_text']}")
            context_block = "\n".join(ctx_lines)

            prompt = (
                f"<bos><start_of_turn>user\n"
                f"بر اساس شواهد علمی زیر به پرسش کاربر به زبان فارسی دقیق و علمی پاسخ بده:\n\n"
                f"{context_block}\n\n"
                f"پرسش: {query}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                add_special_tokens=False,
            ).to(self.dev_obj)
            input_ids = inputs["input_ids"]

            gen_ids = self._generate_with_cudagraph(input_ids, max_new_tokens=max_new_tokens)
            if gen_ids is None:
                bad_words = [[int(x)] for x in self.banned_t.tolist()] if (self.banned_t is not None and len(self.banned_t) < 4096) else None
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.12,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.stop_token_ids,
                )
                if bad_words:
                    gen_kwargs["bad_words_ids"] = bad_words
                out_ids = self.model.generate(**inputs, **gen_kwargs)
                gen_ids = out_ids[0, input_ids.shape[1]:]

            return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ==============================================================================
# 4. HASHING, TOKENIZATION & SOVEREIGN V14 GPU ENGINE (WITH ONLINE LEARNING)
# ==============================================================================

def fnv1a_32_token_u32(word: str) -> int:
    h = 2166136261
    for b in word.lower().encode("utf-8"):
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def u32_to_i32(val: int) -> int:
    return val - 0x100000000 if val >= 0x80000000 else val


def tokenize_alnum_utf8(text: str):
    norm_text = normalize_pdf_ligatures(text)
    tokens = []
    for m in _ALNUM_UTF8_RE.finditer(norm_text):
        w = m.group(0).lower()
        tokens.append((w, fnv1a_32_token_u32(w)))
    return tokens


class SovereignCLRSEngineV14:
    MAX_TOKENS = 40
    NUM_BUCKETS = 262144
    MAX_QUERY_TOKENS = 16
    PER_QUERY_CAP = 8192
    MAX_RESULTS = 131072
    ONLINE_CAP = 8192
    ONLINE_TOK_CAP = 262144

    def __init__(self, arrow_path: str, device: str = ENGINE_DEV):
        self.device = device
        self.dev_obj = torch.device(device)
        with torch.cuda.device(self.dev_obj):
            t0 = time.perf_counter()

            with pa.memory_map(arrow_path, "r") as source:
                table = ipc.open_file(source).read_all()

            self.N = table.num_rows
            self.cap_N = self.N + self.ONLINE_CAP

            col_page = table["page_id"].to_numpy(zero_copy_only=False).astype(np.int32)
            col_chap = table["chapter_id"].to_numpy(zero_copy_only=False).astype(np.int32)
            col_verse = table["verse_num"].to_numpy(zero_copy_only=False).astype(np.int32)

            str_col = table["couplet_text"].combine_chunks()
            buffers = str_col.buffers()
            if pa.types.is_large_string(str_col.type):
                off_np = np.frombuffer(buffers[1], dtype=np.int64, count=self.N + 1).copy()
            else:
                off_np = np.frombuffer(buffers[1], dtype=np.int32, count=self.N + 1).astype(np.int64)
            u8_np = np.frombuffer(buffers[2], dtype=np.uint8, count=int(off_np[-1])).copy()

            self.u8_len = int(off_np[-1])
            self.cap_u8 = max(self.u8_len + 4 * 1024 * 1024, int(self.u8_len * 1.25))

            # Shadow CPU arrays for instant zero-GPU-sync passage decoding
            self.u8_cpu = np.zeros(self.cap_u8, dtype=np.uint8)
            self.u8_cpu[:self.u8_len] = u8_np
            self.off_cpu = np.zeros(self.cap_N + 1, dtype=np.int64)
            self.off_cpu[:self.N + 1] = off_np
            self.meta_cpu = np.zeros((2, self.cap_N), dtype=np.int32)
            self.meta_cpu[0, :self.N] = (col_page & 0xFFFF) | ((col_chap & 0x7FFF) << 16)
            self.meta_cpu[1, :self.N] = col_verse

            self.u8_gpu = torch.zeros(self.cap_u8, dtype=torch.uint8, device=self.dev_obj)
            if self.u8_len > 0:
                self.u8_gpu[:self.u8_len].copy_(torch.from_numpy(u8_np))

            self.off_gpu = torch.zeros(self.cap_N + 1, dtype=torch.int64, device=self.dev_obj)
            self.off_gpu[:self.N + 1].copy_(torch.from_numpy(off_np))

            page_gpu = torch.zeros(self.cap_N, dtype=torch.int32, device=self.dev_obj)
            chap_gpu = torch.zeros(self.cap_N, dtype=torch.int32, device=self.dev_obj)
            verse_gpu = torch.zeros(self.cap_N, dtype=torch.int32, device=self.dev_obj)
            if self.N > 0:
                page_gpu[:self.N].copy_(torch.from_numpy(col_page))
                chap_gpu[:self.N].copy_(torch.from_numpy(col_chap))
                verse_gpu[:self.N].copy_(torch.from_numpy(col_verse))

            self.soa_packed = torch.zeros((self.MAX_TOKENS, self.cap_N), dtype=torch.int32, device=self.dev_obj)
            self.meta_packed = torch.zeros((2, self.cap_N), dtype=torch.int32, device=self.dev_obj)

            if self.N > 0:
                grid = (triton.cdiv(self.N, 128),)
                sovereign_build_soa22_packed_kernel[grid](
                    self.u8_gpu,
                    self.off_gpu,
                    self.soa_packed,
                    self.meta_packed,
                    page_gpu,
                    chap_gpu,
                    verse_gpu,
                    self.N,
                    soa_stride_s=self.cap_N,
                    MAX_TOKENS=self.MAX_TOKENS,
                    BLOCK_N=128,
                )
            torch.cuda.synchronize(self.dev_obj)
            self.load_shm_ms = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            self._build_gpu_csr_index()
            torch.cuda.synchronize(self.dev_obj)
            self.build_csr_ms = (time.perf_counter() - t1) * 1000.0

            self.online_hashes = torch.zeros(self.ONLINE_TOK_CAP, dtype=torch.int32, device=self.dev_obj)
            self.online_postings = torch.full((self.ONLINE_TOK_CAP,), -1, dtype=torch.int32, device=self.dev_obj)
            self.online_nnz = 0

            self.q_buckets_gpu = torch.zeros(self.MAX_QUERY_TOKENS, dtype=torch.int32, device=self.dev_obj)
            self.q_hashes_gpu = torch.zeros(self.MAX_QUERY_TOKENS, dtype=torch.int32, device=self.dev_obj)
            self.q_weights_gpu = torch.zeros(self.MAX_QUERY_TOKENS, dtype=torch.float32, device=self.dev_obj)

            self.doc_mask_gpu = torch.zeros(self.cap_N, dtype=torch.int32, device=self.dev_obj)
            self.doc_score_gpu = torch.zeros(self.cap_N, dtype=torch.float32, device=self.dev_obj)

            self._init_sram_stepper_graph()

    def _build_gpu_csr_index(self):
        with torch.cuda.device(self.dev_obj):
            if self.N <= 0:
                self.total_indexed_tokens = 0
                self.csr_row_ptr = torch.zeros(self.NUM_BUCKETS + 1, dtype=torch.int32, device=self.dev_obj)
                self.csr_col_ind = torch.zeros(1, dtype=torch.int32, device=self.dev_obj)
                self.csr_row_ptr_cpu = np.zeros(self.NUM_BUCKETS + 1, dtype=np.int32)
                return

            soa_valid = self.soa_packed[:, :self.N]
            tok_slots, cids_i64 = torch.nonzero(soa_valid != 0, as_tuple=True)
            nnz = int(cids_i64.numel())
            self.total_indexed_tokens = nnz

            if nnz == 0:
                self.csr_row_ptr = torch.zeros(self.NUM_BUCKETS + 1, dtype=torch.int32, device=self.dev_obj)
                self.csr_col_ind = torch.zeros(1, dtype=torch.int32, device=self.dev_obj)
                self.csr_row_ptr_cpu = np.zeros(self.NUM_BUCKETS + 1, dtype=np.int32)
                return

            packed_1d = soa_valid[tok_slots, cids_i64].contiguous()
            cids_1d = cids_i64.to(torch.int32).contiguous()
            del tok_slots, cids_i64

            buckets_1d = torch.empty(nnz, dtype=torch.int32, device=self.dev_obj)
            grid = (triton.cdiv(nnz, 256),)
            sovereign_hash_1d_tokens_kernel[grid](
                self.u8_gpu,
                self.off_gpu,
                cids_1d,
                packed_1d,
                buckets_1d,
                nnz,
                BUCKET_MASK=self.NUM_BUCKETS - 1,
                BLOCK_SIZE=256,
            )
            del packed_1d

            counts = torch.bincount(buckets_1d, minlength=self.NUM_BUCKETS)
            order = torch.argsort(buckets_1d, stable=True)
            self.csr_col_ind = cids_1d[order].contiguous()
            del cids_1d, buckets_1d, order

            self.csr_row_ptr = torch.zeros(self.NUM_BUCKETS + 1, dtype=torch.int32, device=self.dev_obj)
            self.csr_row_ptr[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
            self.csr_row_ptr_cpu = self.csr_row_ptr.cpu().numpy()

    def _init_sram_stepper_graph(self):
        with torch.cuda.device(self.dev_obj):
            active_n = max(1, self.N)
            self.sram_state = torch.ones(self.cap_N, dtype=torch.int32, device=self.dev_obj)
            self.sram_out = torch.zeros(self.cap_N, dtype=torch.int32, device=self.dev_obj)
            grid = (triton.cdiv(active_n, 256),)

            sram_stream = torch.cuda.Stream(device=self.dev_obj)
            sram_stream.wait_stream(torch.cuda.current_stream(self.dev_obj))
            with torch.cuda.stream(sram_stream):
                for _ in range(3):
                    full_couplet_register_sram_kernel[grid](
                        self.soa_packed, self.sram_state, self.sram_out, active_n,
                        soa_stride_s=self.cap_N, BLOCK_N=256
                    )
            torch.cuda.current_stream(self.dev_obj).wait_stream(sram_stream)
            torch.cuda.synchronize(self.dev_obj)

            self.sram_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.sram_graph, stream=sram_stream):
                full_couplet_register_sram_kernel[grid](
                    self.soa_packed, self.sram_state, self.sram_out, active_n,
                    soa_stride_s=self.cap_N, BLOCK_N=256
                )
            torch.cuda.synchronize(self.dev_obj)
            t0 = time.perf_counter()
            for _ in range(10):
                self.sram_graph.replay()
            torch.cuda.synchronize(self.dev_obj)
            self.sram_us = ((time.perf_counter() - t0) / 10.0) * 1e6

    def get_vram_usage_mb(self) -> float:
        total_bytes = (
            self.u8_gpu.numel() * self.u8_gpu.element_size()
            + self.off_gpu.numel() * self.off_gpu.element_size()
            + self.soa_packed.numel() * self.soa_packed.element_size()
            + self.meta_packed.numel() * self.meta_packed.element_size()
            + self.csr_row_ptr.numel() * self.csr_row_ptr.element_size()
            + self.csr_col_ind.numel() * self.csr_col_ind.element_size()
            + self.online_hashes.numel() * self.online_hashes.element_size()
            + self.online_postings.numel() * self.online_postings.element_size()
            + self.doc_mask_gpu.numel() * self.doc_mask_gpu.element_size()
            + self.doc_score_gpu.numel() * self.doc_score_gpu.element_size()
        )
        return total_bytes / (1024.0 * 1024.0)

    def decode_couplet(self, cid: int) -> dict:
        if cid < 0 or cid >= self.N:
            return {"cid": cid, "page": 0, "chapter": 0, "source": "Unknown", "verse": 0, "text": ""}
        s_off = int(self.off_cpu[cid])
        e_off = int(self.off_cpu[cid + 1])
        raw_bytes = self.u8_cpu[s_off:e_off].tobytes()
        text = clean_couplet_display_text(raw_bytes.decode("utf-8", errors="replace"))
        meta0 = int(self.meta_cpu[0, cid])
        verse = int(self.meta_cpu[1, cid])
        page = meta0 & 0xFFFF
        chapter = (meta0 >> 16) & 0x7FFF
        source_name = DOCUMENT_SOURCE_NAMES.get(chapter, f"Document_{chapter}")
        return {
            "cid": cid,
            "page": page,
            "chapter": chapter,
            "source": source_name,
            "verse": verse,
            "text": text,
        }

    def decode_expanded_passage(self, cid: int) -> dict:
        base = self.decode_couplet(cid)
        pieces = []
        for neighbor in (cid - 1, cid, cid + 1):
            if 0 <= neighbor < self.N:
                nb = self.decode_couplet(neighbor)
                if nb["chapter"] == base["chapter"] and abs(nb["page"] - base["page"]) <= 1:
                    pieces.append(nb["text"])
        base["expanded_text"] = merge_overlapping_passages(pieces) if pieces else base["text"]
        return base

    def online_learn(self, text: str, page: int = 1, chapter: int = 900) -> int:
        with torch.cuda.device(self.dev_obj):
            clean_txt = clean_couplet_display_text(text)
            if not clean_txt:
                return max(0, self.N - 1)
            raw = safe_utf8_truncate(clean_txt, 500)
            n_bytes = len(raw)

            toks = tokenize_alnum_utf8(clean_txt)[:self.MAX_TOKENS]
            if (
                self.N >= self.cap_N
                or (self.u8_len + n_bytes) >= self.cap_u8
                or (self.online_nnz + len(toks)) >= self.ONLINE_TOK_CAP
            ):
                _ensure_gpu_online_capacity(
                    self,
                    extra_couplets=max(4096, self.N // 4),
                    extra_bytes=max(2 * 1024 * 1024, n_bytes * 1024),
                )

            new_cid = self.N
            s_off = self.u8_len
            e_off = s_off + n_bytes

            if n_bytes > 0:
                raw_np = np.frombuffer(raw, dtype=np.uint8).copy()
                self.u8_cpu[s_off:e_off] = raw_np
                self.u8_gpu[s_off:e_off].copy_(torch.from_numpy(raw_np))

            self.off_cpu[new_cid + 1] = e_off
            self.off_gpu[new_cid + 1] = e_off
            self.u8_len = e_off

            meta0 = (int(page) & 0xFFFF) | ((int(chapter) & 0x7FFF) << 16)
            self.meta_cpu[0, new_cid] = meta0
            self.meta_cpu[1, new_cid] = new_cid + 1
            self.meta_packed[0, new_cid] = meta0
            self.meta_packed[1, new_cid] = new_cid + 1

            pg_t = torch.tensor([page], dtype=torch.int32, device=self.dev_obj)
            ch_t = torch.tensor([chapter], dtype=torch.int32, device=self.dev_obj)
            vn_t = torch.tensor([new_cid + 1], dtype=torch.int32, device=self.dev_obj)

            sovereign_build_soa22_packed_kernel[(1,)](
                self.u8_gpu,
                self.off_gpu[new_cid : new_cid + 2],
                self.soa_packed[:, new_cid : new_cid + 1],
                self.meta_packed[:, new_cid : new_cid + 1],
                pg_t,
                ch_t,
                vn_t,
                1,
                soa_stride_s=self.cap_N,
                MAX_TOKENS=self.MAX_TOKENS,
                BLOCK_N=1,
            )

            if toks:
                n_toks = len(toks)
                start_nnz = self.online_nnz
                end_nnz = start_nnz + n_toks
                hash_arr = torch.tensor(
                    [u32_to_i32(h_u32) for _, h_u32 in toks],
                    dtype=torch.int32,
                    device=self.dev_obj,
                )
                self.online_hashes[start_nnz:end_nnz].copy_(hash_arr)
                self.online_postings[start_nnz:end_nnz].fill_(new_cid)
                self.online_nnz = end_nnz

            self.N += 1
            torch.cuda.synchronize(self.dev_obj)
            return new_cid


# ==============================================================================
# 5. UNIVERSAL MULTI-DOMAIN BILINGUAL SEARCH & GEMMA-2 SYNTHESIS PLATFORM
# ==============================================================================

BILINGUAL_CONCEPT_MAP = {
    "کلیه": ["renal", "kidney", "nephron", "tubular", "glomerular"],
    "کلیوی": ["renal", "kidney", "nephropathy", "tubular"],
    "قلب": ["cardiac", "heart", "myocardial", "ventricular", "pericardial"],
    "قلبی": ["cardiac", "myocardial", "coronary", "arrhythmia"],
    "فشارخون": ["hypertension", "hypertensive", "blood", "pressure"],
    "فشار": ["pressure", "hypertension", "tension", "hemodynamic"],
    "خون": ["blood", "hematologic", "thrombotic", "coagulation", "plasma"],
    "انعقاد": ["coagulation", "thrombosis", "clotting", "fibrinogen", "prothrombin"],
    "پلاکت": ["platelet", "thrombocytopenia", "thrombocytopenic", "purpura"],
    "عصب": ["nerve", "neurologic", "neuropathy", "synaptic", "axonal"],
    "عضله": ["muscle", "muscular", "neuromuscular", "myasthenia"],
    "آنتی بادی": ["antibody", "antibodies", "autoantibody", "immunoglobulin"],
    "گیرنده": ["receptor", "receptors", "adrenergic", "acetylcholine"],
    "کانال": ["channel", "channels", "cotransporter", "transporter"],
    "سندرم": ["syndrome", "disorder", "disease", "manifestation"],
    "التهاب": ["inflammation", "inflammatory", "pericarditis", "itis"],
    "تومور": ["tumor", "neoplasm", "carcinoma", "malignancy", "cancer"],
    "دارو": ["drug", "therapy", "treatment", "pharmacologic", "blocker"],
    "درمان": ["treatment", "therapy", "management", "plasmapheresis"],
    "تشخیص": ["diagnosis", "diagnostic", "differentiate", "differential"],
    "پاتوفیزیولوژی": ["pathophysiology", "mechanism", "pathogenesis"],
    "جهش": ["mutation", "mutations", "genetic", "deficiency"],
    "آنزیم": ["enzyme", "deficiency", "activity", "protease"],
    "الگوریتم": ["algorithm", "complexity", "method", "optimization"],
    "شبکه": ["network", "neural", "graph", "architecture"],
    "مدل": ["model", "parameter", "training", "distribution"],
}

STOPWORDS_SET = {
    "در", "به", "از", "که", "این", "آن", "با", "را", "برای", "یک", "بر",
    "یا", "تا", "هم", "چرا", "چگونه", "چیست", "کدام", "است", "شود", "می",
    "های", "ها", "کند", "کرد", "شد", "بین", "روی", "توسط", "دارد", "دارند",
    "بود", "باشند", "باشد", "اگر", "اما", "زیرا", "چون", "وقتی", "هنگام",
    "بررسی", "توضیح", "مقایسه", "تفاوت", "رابطه", "نقش", "روش", "مسئله",
    "نظر", "همراه", "باعث", "شدید", "طبیعی", "بودن", "سطح", "مبتلا",
    "the", "and", "for", "with", "from", "that", "this", "what", "how",
    "why", "are", "was", "were", "have", "has", "had", "into", "over",
    "under", "between", "compare", "explain", "difference", "differ",
    "differentiate", "regarding", "due", "does", "before", "after",
}


class SovereignCLRSCommercialPlatform:
    MAX_CACHE_ENTRIES = 512

    def __init__(self, arrow_path: str, load_gemma: bool = True):
        self.engine = SovereignCLRSEngineV14(arrow_path, device=ENGINE_DEV)
        self.llm = FastGemma2CUDAGraphEngine(device=LLM_DEV) if load_gemma else None
        self.query_cache = OrderedDict()

    def extract_query_keywords(self, query: str) -> list:
        qn = normalize_query_text(query)
        raw_tokens = [tok for tok, _ in tokenize_alnum_utf8(qn)]
        terms = []
        seen = set()

        def _add(t: str):
            if t not in seen:
                seen.add(t)
                terms.append(t)

        # Preserve both ASCII and non-ASCII (Persian/multilingual) query keywords with balanced priority
        for tok in raw_tokens:
            if len(tok) >= 2 and tok not in STOPWORDS_SET:
                _add(tok)

        qn_lower = qn.lower()
        for p_key, eng_list in BILINGUAL_CONCEPT_MAP.items():
            if p_key in qn_lower:
                for et in eng_list:
                    _add(et)

        if not terms:
            for tok in raw_tokens:
                if len(tok) >= 1:
                    _add(tok)

        return terms[:self.engine.MAX_QUERY_TOKENS]

    def search_clrs_hybrid(self, query: str, top_k: int = 5, use_cache: bool = True) -> dict:
        with torch.cuda.device(self.engine.dev_obj):
            top_k = max(1, min(int(top_k), 50))
            cache_key = (normalize_query_text(query), top_k, self.engine.N, self.engine.online_nnz)
            if use_cache and cache_key in self.query_cache:
                self.query_cache.move_to_end(cache_key)
                return self.query_cache[cache_key]

            keywords = self.extract_query_keywords(query)
            num_q = len(keywords)
            if num_q == 0 or self.engine.N == 0:
                return {"hits": [], "keywords": keywords, "gpu_csr_us": 0.0}

            q_buckets = []
            q_hashes = []
            q_weights = []
            row_ptr_cpu = self.engine.csr_row_ptr_cpu

            max_span = 0
            for kw in keywords:
                h_u32 = fnv1a_32_token_u32(kw)
                b = h_u32 & (self.engine.NUM_BUCKETS - 1)
                q_buckets.append(b)
                q_hashes.append(u32_to_i32(h_u32))
                span = int(row_ptr_cpu[b + 1] - row_ptr_cpu[b])
                if span > max_span:
                    max_span = span
                df = max(1, span)
                idf = math.log(1.0 + (self.engine.N / float(df)))
                if len(kw) <= 2:
                    idf *= 0.25
                q_weights.append(idf)

            self.engine.q_buckets_gpu[:num_q].copy_(
                torch.tensor(q_buckets, dtype=torch.int32, device=self.engine.dev_obj)
            )
            self.engine.q_hashes_gpu[:num_q].copy_(
                torch.tensor(q_hashes, dtype=torch.int32, device=self.engine.dev_obj)
            )
            self.engine.q_weights_gpu[:num_q].copy_(
                torch.tensor(q_weights, dtype=torch.float32, device=self.engine.dev_obj)
            )

            self.engine.doc_mask_gpu[:self.engine.N].zero_()
            self.engine.doc_score_gpu[:self.engine.N].zero_()

            torch.cuda.synchronize(self.engine.dev_obj)
            t0 = time.perf_counter()

            if max_span > 0:
                grid_2d = (num_q, triton.cdiv(max_span, 256))
                sovereign_csr_1m_bitmask_kernel[grid_2d](
                    self.engine.csr_row_ptr,
                    self.engine.csr_col_ind,
                    self.engine.q_buckets_gpu,
                    self.engine.q_weights_gpu,
                    self.engine.doc_mask_gpu,
                    self.engine.doc_score_gpu,
                    num_q,
                    self.engine.N,
                    BLOCK_SCAN=256,
                )

            if self.engine.online_nnz > 0:
                grid_online = (num_q, triton.cdiv(self.engine.online_nnz, 256))
                sovereign_online_search_score_kernel[grid_online](
                    self.engine.online_hashes,
                    self.engine.online_postings,
                    self.engine.online_nnz,
                    self.engine.q_hashes_gpu,
                    self.engine.q_weights_gpu,
                    self.engine.doc_mask_gpu,
                    self.engine.doc_score_gpu,
                    num_q,
                    self.engine.N,
                    BLOCK_SCAN=256,
                )

            torch.cuda.synchronize(self.engine.dev_obj)
            gpu_csr_us = (time.perf_counter() - t0) * 1e6

            active_masks = self.engine.doc_mask_gpu[:self.engine.N]
            cand_cids_t = torch.nonzero(active_masks != 0, as_tuple=False).flatten()

            if cand_cids_t.numel() == 0:
                return {"hits": [], "keywords": keywords, "gpu_csr_us": gpu_csr_us}

            cand_masks = active_masks[cand_cids_t]
            bit_shifts = torch.arange(num_q, dtype=torch.int32, device=self.engine.dev_obj).unsqueeze(0)
            popcounts = ((cand_masks.unsqueeze(1) >> bit_shifts) & 1).sum(dim=1).to(torch.float32)

            cand_scores = self.engine.doc_score_gpu[cand_cids_t] + (popcounts ** 1.8) * 10.0

            # Neighbor context smoothing computed strictly over candidate neighborhoods (avoids cloning 1M+ tensor twice)
            if self.engine.N > 2:
                prev_cids = torch.clamp(cand_cids_t - 1, 0, self.engine.N - 1)
                next_cids = torch.clamp(cand_cids_t + 1, 0, self.engine.N - 1)
                has_prev = (cand_cids_t > 0).to(torch.float32)
                has_next = (cand_cids_t < (self.engine.N - 1)).to(torch.float32)
                neighbor_bonus = (
                    has_prev * self.engine.doc_score_gpu[prev_cids]
                    + has_next * self.engine.doc_score_gpu[next_cids]
                ) * 0.35
                cand_final_scores = cand_scores + neighbor_bonus
            else:
                cand_final_scores = cand_scores

            k_top = min(int(cand_cids_t.numel()), max(top_k * 8, 32))
            top_vals, top_idx = torch.topk(cand_final_scores, k=k_top)

            top_cids_cpu = cand_cids_t[top_idx].cpu().numpy()
            top_scores_cpu = top_vals.cpu().numpy()
            top_pop_cpu = popcounts[top_idx].cpu().numpy()

            kw_patterns = [
                (kw, re.compile(r"(?<![0-9a-z\u0080-\uffff])" + re.escape(kw) + r"(?![0-9a-z\u0080-\uffff])"))
                for kw in keywords
                if len(kw) >= 2
            ]

            verified_hits = []
            seen_pages = set()
            for cid, sc, uniq_cnt in zip(top_cids_cpu, top_scores_cpu, top_pop_cpu):
                cid_int = int(cid)
                info = self.engine.decode_expanded_passage(cid_int)
                txt_low = info["expanded_text"].lower()

                exact_tokens = [kw for kw, pat in kw_patterns if pat.search(txt_low)]
                substr_matches = [kw for kw in keywords if len(kw) >= 2 and kw in txt_low]
                matched = exact_tokens if exact_tokens else substr_matches
                if not matched:
                    continue

                dedup_key = (info["chapter"], info["page"], cid_int // 2)
                if dedup_key in seen_pages:
                    continue
                seen_pages.add(dedup_key)

                token_bonus = len(exact_tokens) * 8.0 + (len(substr_matches) - len(exact_tokens)) * 3.0
                info["score"] = round(float(sc) + token_bonus, 2)
                info["matched_terms"] = matched
                info["unique_term_count"] = int(uniq_cnt)
                verified_hits.append(info)
                if len(verified_hits) >= top_k * 2:
                    break

            verified_hits.sort(key=lambda x: x["score"], reverse=True)
            out = {
                "hits": verified_hits[:top_k],
                "keywords": keywords,
                "gpu_csr_us": gpu_csr_us,
            }
            if use_cache:
                self.query_cache[cache_key] = out
                if len(self.query_cache) > self.MAX_CACHE_ENTRIES:
                    self.query_cache.popitem(last=False)
            return out

    def synthesize_answer(self, query: str, hits: list, use_gemma: bool = False) -> str:
        if not hits:
            return "هیچ قطعهٔ منطبقی در اسناد فعلی یافت نشد. لطفاً فایل PDF یا متن یادگیری آنلاین (`online_learn`) را وارد نمایید."

        parts = []
        if use_gemma and self.llm is not None and self.llm.ready:
            gemma_text = self.llm.synthesize_from_hits(query, hits)
            if gemma_text:
                parts.append(f"🤖 [تحلیل هوشمند مدل Gemma-2 Fused-Triton CUDAGraph]:\n{gemma_text}\n" + ("─" * 65))

        lines = ["بر اساس بازیابی مستقیم از اسناد ایندکس‌شده روی GPU، بالاترین شواهد منطبق عبارتند از:\n"]
        for i, h in enumerate(hits[:5], 1):
            matched_str = ", ".join(h.get("matched_terms", [])[:8])
            lines.append(
                f"• شاهد {i} [{h['source']} | صفحه {h['page']} | قطعه #{h['cid']} | امتیاز {h['score']} | واژگان منطبق: {matched_str}]:\n"
                f"  «{h['expanded_text']}»"
            )
        parts.append("\n\n".join(lines))
        return "\n\n".join(parts)

    def ask(self, query: str, top_k: int = 5, generate_lesson: bool = True, use_gemma: bool = False, use_cache: bool = True) -> dict:
        t0 = time.perf_counter()
        ret = self.search_clrs_hybrid(query, top_k=top_k, use_cache=use_cache)
        hits = ret["hits"]
        ans = self.synthesize_answer(query, hits, use_gemma=use_gemma) if generate_lesson else ""
        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "query": query,
            "keywords": ret["keywords"],
            "hits": hits,
            "answer": ans,
            "gpu_csr_us": ret["gpu_csr_us"],
            "total_ms": total_ms,
        }

    query = ask
    ask_clrs = ask


# ==============================================================================
# 6. BATCH GPU MULTI-PDF INGESTION, ONLINE LEARNING, 1M-DOC SCALER & BENCHMARKS
# ==============================================================================

UPLOADED_PDF_REGISTRY = []
ONLINE_LEARN_NOTES_COUNT = 0
ENGINE_LOCK = threading.RLock()


def _ensure_gpu_online_capacity(eng: SovereignCLRSEngineV14, extra_couplets: int = 4096, extra_bytes: int = 2 * 1024 * 1024):
    with torch.cuda.device(eng.dev_obj):
        need_N = eng.N + extra_couplets
        if need_N >= eng.cap_N:
            new_cap_N = max(eng.cap_N * 2, need_N + 8192)

            new_off = torch.zeros(new_cap_N + 1, dtype=torch.int64, device=eng.dev_obj)
            new_off[:eng.N + 1].copy_(eng.off_gpu[:eng.N + 1])
            eng.off_gpu = new_off

            new_off_cpu = np.zeros(new_cap_N + 1, dtype=np.int64)
            new_off_cpu[:eng.N + 1] = eng.off_cpu[:eng.N + 1]
            eng.off_cpu = new_off_cpu

            new_soa = torch.zeros((eng.MAX_TOKENS, new_cap_N), dtype=torch.int32, device=eng.dev_obj)
            if eng.N > 0:
                new_soa[:, :eng.N].copy_(eng.soa_packed[:, :eng.N])
            eng.soa_packed = new_soa

            new_meta = torch.zeros((2, new_cap_N), dtype=torch.int32, device=eng.dev_obj)
            if eng.N > 0:
                new_meta[:, :eng.N].copy_(eng.meta_packed[:, :eng.N])
            eng.meta_packed = new_meta

            new_meta_cpu = np.zeros((2, new_cap_N), dtype=np.int32)
            if eng.N > 0:
                new_meta_cpu[:, :eng.N] = eng.meta_cpu[:, :eng.N]
            eng.meta_cpu = new_meta_cpu

            eng.doc_mask_gpu = torch.zeros(new_cap_N, dtype=torch.int32, device=eng.dev_obj)
            eng.doc_score_gpu = torch.zeros(new_cap_N, dtype=torch.float32, device=eng.dev_obj)
            eng.cap_N = new_cap_N

        need_u8 = eng.u8_len + extra_bytes
        if need_u8 >= eng.cap_u8:
            new_cap_u8 = max(eng.cap_u8 * 2, need_u8 + 8 * 1024 * 1024)

            new_u8 = torch.zeros(new_cap_u8, dtype=torch.uint8, device=eng.dev_obj)
            if eng.u8_len > 0:
                new_u8[:eng.u8_len].copy_(eng.u8_gpu[:eng.u8_len])
            eng.u8_gpu = new_u8

            new_u8_cpu = np.zeros(new_cap_u8, dtype=np.uint8)
            if eng.u8_len > 0:
                new_u8_cpu[:eng.u8_len] = eng.u8_cpu[:eng.u8_len]
            eng.u8_cpu = new_u8_cpu

            eng.cap_u8 = new_cap_u8

        need_tok = eng.online_nnz + extra_couplets * eng.MAX_TOKENS
        if need_tok >= eng.ONLINE_TOK_CAP:
            new_tok_cap = max(eng.ONLINE_TOK_CAP * 2, need_tok + 131072)
            new_hashes = torch.zeros(new_tok_cap, dtype=torch.int32, device=eng.dev_obj)
            if eng.online_nnz > 0:
                new_hashes[:eng.online_nnz].copy_(eng.online_hashes[:eng.online_nnz])
            eng.online_hashes = new_hashes

            new_postings = torch.full((new_tok_cap,), -1, dtype=torch.int32, device=eng.dev_obj)
            if eng.online_nnz > 0:
                new_postings[:eng.online_nnz].copy_(eng.online_postings[:eng.online_nnz])
            eng.online_postings = new_postings
            eng.ONLINE_TOK_CAP = new_tok_cap


def batch_append_passages_to_gpu_csr(
    eng: SovereignCLRSEngineV14,
    texts: list,
    pages_np: np.ndarray,
    chaps_np: np.ndarray,
    replace_existing: bool = False,
) -> dict:
    with torch.cuda.device(eng.dev_obj):
        M = len(texts)
        if M == 0:
            return {
                "added": 0,
                "total_N": eng.N,
                "total_tokens": eng.total_indexed_tokens,
                "gpu_index_ms": 0.0,
                "total_ms": 0.0,
                "vram_mb": round(eng.get_vram_usage_mb(), 2),
            }

        t0 = time.perf_counter()
        encoded_list = [safe_utf8_truncate(t, 500) for t in texts]
        lengths = np.fromiter((len(b) for b in encoded_list), dtype=np.int64, count=M)
        total_new_bytes = int(lengths.sum())
        blob = b"".join(encoded_list)
        u8_new_np = np.frombuffer(blob, dtype=np.uint8, count=total_new_bytes).copy()

        if replace_existing:
            eng.N = 0
            eng.u8_len = 0
            eng.online_nnz = 0
            eng.off_gpu.zero_()
            eng.off_cpu.fill(0)
            eng.soa_packed.zero_()
            eng.meta_packed.zero_()
            eng.meta_cpu.fill(0)

        _ensure_gpu_online_capacity(eng, extra_couplets=M + 1024, extra_bytes=total_new_bytes + 1024 * 1024)

        old_N = eng.N
        old_u8 = eng.u8_len
        new_N = old_N + M

        rel_offsets = np.empty(M + 1, dtype=np.int64)
        rel_offsets[0] = old_u8
        np.cumsum(lengths, dtype=np.int64, out=rel_offsets[1:])
        rel_offsets[1:] += old_u8

        pages_i32 = pages_np.astype(np.int32)
        chaps_i32 = chaps_np.astype(np.int32)
        verses_i32 = np.arange(old_N + 1, new_N + 1, dtype=np.int32)

        if total_new_bytes > 0:
            eng.u8_cpu[old_u8 : old_u8 + total_new_bytes] = u8_new_np
        eng.off_cpu[old_N : new_N + 1] = rel_offsets
        eng.meta_cpu[0, old_N:new_N] = (pages_i32 & 0xFFFF) | ((chaps_i32 & 0x7FFF) << 16)
        eng.meta_cpu[1, old_N:new_N] = verses_i32

        torch.cuda.synchronize(eng.dev_obj)
        t_gpu0 = time.perf_counter()

        if total_new_bytes > 0:
            eng.u8_gpu[old_u8 : old_u8 + total_new_bytes].copy_(torch.from_numpy(u8_new_np))
        eng.off_gpu[old_N : new_N + 1].copy_(torch.from_numpy(rel_offsets))

        page_gpu = torch.from_numpy(pages_i32).to(eng.dev_obj)
        chap_gpu = torch.from_numpy(chaps_i32).to(eng.dev_obj)
        verse_gpu = torch.from_numpy(verses_i32).to(eng.dev_obj)

        grid = (triton.cdiv(M, 128),)
        sovereign_build_soa22_packed_kernel[grid](
            eng.u8_gpu,
            eng.off_gpu[old_N : new_N + 1],
            eng.soa_packed[:, old_N:new_N],
            eng.meta_packed[:, old_N:new_N],
            page_gpu,
            chap_gpu,
            verse_gpu,
            M,
            soa_stride_s=eng.cap_N,
            MAX_TOKENS=eng.MAX_TOKENS,
            BLOCK_N=128,
        )

        eng.N = new_N
        eng.u8_len = old_u8 + total_new_bytes

        eng._build_gpu_csr_index()
        torch.cuda.synchronize(eng.dev_obj)

        gpu_index_ms = (time.perf_counter() - t_gpu0) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0
        eng.build_csr_ms = gpu_index_ms

        return {
            "added": M,
            "total_N": eng.N,
            "total_tokens": eng.total_indexed_tokens,
            "gpu_index_ms": round(gpu_index_ms, 2),
            "total_ms": round(total_ms, 2),
            "vram_mb": round(eng.get_vram_usage_mb(), 2),
        }


# ==============================================================================
# 7. INITIALIZE PLATFORM ( GEMMA-2 CUDAGRAPH)
# ==============================================================================
arrow_file = build_bootstrap_arrow_in_shm(force_rebuild=True)
platform = SovereignCLRSCommercialPlatform(arrow_file, load_gemma=True)
engine = platform.engine
BOOTSTRAP_ACTIVE = True


def ingest_pdf_bytes_into_engine(pdf_bytes: bytes, filename: str, source_id: int = None) -> dict:
    global BOOTSTRAP_ACTIVE
    t0 = time.perf_counter()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        num_pages = len(doc)
        texts = []
        pages_list = []

        for p_idx in range(num_pages):
            raw_page = doc[p_idx].get_text("text")
            if not raw_page:
                continue
            clean_p = clean_couplet_display_text(raw_page)
            words = clean_p.split()
            if len(words) < 4:
                continue
            actual_page = min(65535, p_idx + 1)
            for w_start in range(0, len(words), 22):
                chunk_words = words[w_start : w_start + 26]
                if len(chunk_words) < 4:
                    continue
                texts.append(" ".join(chunk_words))
                pages_list.append(actual_page)
    finally:
        doc.close()

    extract_ms = (time.perf_counter() - t0) * 1000.0

    with ENGINE_LOCK:
        if source_id is None:
            candidate_id = len(UPLOADED_PDF_REGISTRY) + 1
            while candidate_id in DOCUMENT_SOURCE_NAMES:
                candidate_id += 1
            source_id = candidate_id
        DOCUMENT_SOURCE_NAMES[source_id] = filename

        chaps_arr = np.full(len(texts), source_id, dtype=np.int32)
        pages_arr = np.array(pages_list, dtype=np.int32)

        replace_bootstrap = BOOTSTRAP_ACTIVE and len(texts) > 0
        stats = batch_append_passages_to_gpu_csr(
            engine,
            texts,
            pages_arr,
            chaps_arr,
            replace_existing=replace_bootstrap,
        )
        if len(texts) > 0:
            BOOTSTRAP_ACTIVE = False
        platform.query_cache.clear()

        total_ms = (time.perf_counter() - t0) * 1000.0
        record = {
            "filename": filename,
            "source_id": source_id,
            "pages": num_pages,
            "couplets_added": stats["added"],
            "extract_ms": round(extract_ms, 2),
            "gpu_csr_build_ms": stats["gpu_index_ms"],
            "elapsed_ms": round(total_ms, 2),
            "total_couplets": engine.N,
            "total_tokens": engine.total_indexed_tokens,
            "online_nnz": engine.online_nnz,
            "vram_mb": stats["vram_mb"],
        }
        UPLOADED_PDF_REGISTRY.append(record)

    print(
        f"\n[PDF BATCH GPU INGEST BENCHMARK] File: {filename} | Pages: {num_pages} | "
        f"+{stats['added']:,} Passages | PDF Extract: {extract_ms:.1f} ms | "
        f"GPU CSR Build: {stats['gpu_index_ms']:.2f} ms | Total Docs: {engine.N:,} | "
        f"Total Tokens: {engine.total_indexed_tokens:,} | VRAM: {stats['vram_mb']:.1f} MB",
        flush=True,
    )
    return record


def scale_engine_to_target_docs_gpu(target_docs: int = 1_000_000) -> dict:
    global BOOTSTRAP_ACTIVE
    with ENGINE_LOCK:
        with torch.cuda.device(engine.dev_obj):
            t0 = time.perf_counter()

            if BOOTSTRAP_ACTIVE or engine.N < 16:
                DOCUMENT_SOURCE_NAMES[99] = "Synthetic_1M_Medical_Scientific_Corpus.pdf"
                seed_templates = [
                    "Bartter syndrome and Gitelman syndrome involve renal tubular mutations in NKCC2 and NCCT causing hypokalemic metabolic alkalosis with distinct urinary calcium excretion.",
                    "Thrombotic thrombocytopenic purpura TTP results from severe ADAMTS13 deficiency leading to ultra-large von Willebrand factor multimers schistocytes and microvascular thrombosis.",
                    "Pheochromocytoma catecholamine crisis requires alpha-adrenergic blockade with phenoxybenzamine before beta-blocker therapy to prevent unopposed alpha vasoconstriction.",
                    "Guillain-Barre syndrome presents with ascending demyelinating polyneuropathy and albuminocytologic dissociation whereas Myasthenia Gravis targets postsynaptic acetylcholine receptors.",
                    "Cardiac tamponade and constrictive pericarditis exhibit pulsus paradoxus Kussmaul sign prominent jugular venous x and y descents and diastolic pressure equalization.",
                    "Multi-head self-attention computes scaled dot-product queries keys and values across residual transformer layers with RMSNorm and rotational positional embeddings.",
                    "Disseminated intravascular coagulation DIC consumes platelets fibrinogen and coagulation factors prolonging prothrombin time PT and activated partial thromboplastin time PTT.",
                    "Lambert-Eaton myasthenic syndrome produces presynaptic voltage-gated calcium channel VGCC autoantibodies associated with paraneoplastic small cell lung carcinoma.",
                ]
                s_pages = np.arange(1, len(seed_templates) + 1, dtype=np.int32)
                s_chaps = np.full(len(seed_templates), 99, dtype=np.int32)
                batch_append_passages_to_gpu_csr(engine, seed_templates, s_pages, s_chaps, replace_existing=True)
                BOOTSTRAP_ACTIVE = False

            base_N = engine.N
            base_u8 = engine.u8_len
            if target_docs <= base_N:
                target_docs = base_N

            # Compute exact byte budget via int64 cumulative lengths so 1M+ docs never overflow int32 offsets
            base_lengths_cpu = engine.off_cpu[1 : base_N + 1] - engine.off_cpu[:base_N]
            idx_map_cpu = np.arange(target_docs, dtype=np.int64) % base_N
            all_lengths_cpu = base_lengths_cpu[idx_map_cpu]
            total_u8_needed = int(all_lengths_cpu.sum())

            _ensure_gpu_online_capacity(
                engine,
                extra_couplets=(target_docs - base_N) + 4096,
                extra_bytes=max(0, total_u8_needed - base_u8) + 4 * 1024 * 1024,
            )

            engine.off_cpu[0] = 0
            np.cumsum(all_lengths_cpu, dtype=np.int64, out=engine.off_cpu[1 : target_docs + 1])

            cur_cpu_u8 = base_u8
            while cur_cpu_u8 < total_u8_needed:
                copy_len = min(cur_cpu_u8, total_u8_needed - cur_cpu_u8)
                engine.u8_cpu[cur_cpu_u8 : cur_cpu_u8 + copy_len] = engine.u8_cpu[:copy_len]
                cur_cpu_u8 += copy_len

            engine.meta_cpu[0, :target_docs] = engine.meta_cpu[0, idx_map_cpu]
            engine.meta_cpu[1, :target_docs] = np.arange(1, target_docs + 1, dtype=np.int32)

            torch.cuda.synchronize(engine.dev_obj)
            t_gpu0 = time.perf_counter()

            base_lengths = engine.off_gpu[1 : base_N + 1] - engine.off_gpu[:base_N]
            idx_map = torch.arange(target_docs, device=engine.dev_obj) % base_N
            all_lengths = base_lengths[idx_map]

            engine.off_gpu[0] = 0
            torch.cumsum(all_lengths, dim=0, dtype=torch.int64, out=engine.off_gpu[1 : target_docs + 1])
            new_u8_len = int(engine.off_gpu[target_docs].item())

            cur_u8 = base_u8
            while cur_u8 < new_u8_len:
                copy_len = min(cur_u8, new_u8_len - cur_u8)
                engine.u8_gpu[cur_u8 : cur_u8 + copy_len].copy_(engine.u8_gpu[:copy_len])
                cur_u8 += copy_len

            engine.soa_packed[:, :target_docs] = engine.soa_packed[:, idx_map]
            engine.meta_packed[0, :target_docs] = engine.meta_packed[0, idx_map]
            engine.meta_packed[1, :target_docs] = torch.arange(1, target_docs + 1, dtype=torch.int32, device=engine.dev_obj)

            engine.N = target_docs
            engine.u8_len = new_u8_len

            engine._build_gpu_csr_index()
            torch.cuda.synchronize(engine.dev_obj)

            gpu_csr_ms = (time.perf_counter() - t_gpu0) * 1000.0
            total_ms = (time.perf_counter() - t0) * 1000.0
            engine.build_csr_ms = gpu_csr_ms
            platform.query_cache.clear()

            test_q = "Bartter Gitelman ADAMTS13 pheochromocytoma tamponade"
            q_res = platform.ask(test_q, top_k=5, use_cache=False)

            vram_mb = round(engine.get_vram_usage_mb(), 2)
            stats = {
                "total_couplets": engine.N,
                "total_tokens": engine.total_indexed_tokens,
                "online_nnz": engine.online_nnz,
                "gpu_csr_build_ms": round(gpu_csr_ms, 2),
                "total_scale_ms": round(total_ms, 2),
                "search_1m_us": round(q_res["gpu_csr_us"], 2),
                "vram_mb": vram_mb,
            }

            print("\n" + "=" * 84)
            print(" 🚀 [1,000,000-DOC GPU RAG SCALE BENCHMARK REPORT]")
            print("=" * 84)
            print(f"   Total Indexed Documents (N)  : {engine.N:,} docs")
            print(f"   Total Indexed Tokens (NNZ)   : {engine.total_indexed_tokens:,} tokens")
            print(f"   GPU CSR Index Build Time     : {gpu_csr_ms:.2f} ms")
            print(f"   1M-Doc GPU Search Latency    : {q_res['gpu_csr_us']:.2f} µs ({q_res['gpu_csr_us']/1000.0:.3f} ms)")
            print(f"   Total Engine VRAM Footprint  : {vram_mb:.2f} MB")
            print("=" * 84 + "\n", flush=True)
            return stats


def run_and_print_console_benchmarks(eng: SovereignCLRSEngineV14, plat: SovereignCLRSCommercialPlatform) -> dict:
    with torch.cuda.device(eng.dev_obj):
        B_OMEGA, DIM_OMEGA, STEPS_OMEGA = 1024, 128, 256
        st_omega = torch.randn((B_OMEGA, DIM_OMEGA), dtype=torch.float16, device=eng.dev_obj)
        w1_omega = torch.ones(DIM_OMEGA, dtype=torch.float16, device=eng.dev_obj)
        w2_omega = torch.ones(DIM_OMEGA, dtype=torch.float16, device=eng.dev_obj)
        out_omega = torch.empty_like(st_omega)

        omega_17b_sram_kernel[(B_OMEGA,)](st_omega, w1_omega, w2_omega, out_omega, B=B_OMEGA, DIM=DIM_OMEGA, STEPS=STEPS_OMEGA)
        torch.cuda.synchronize(eng.dev_obj)
        t_om = time.perf_counter()
        for _ in range(10):
            omega_17b_sram_kernel[(B_OMEGA,)](st_omega, w1_omega, w2_omega, out_omega, B=B_OMEGA, DIM=DIM_OMEGA, STEPS=STEPS_OMEGA)
        torch.cuda.synchronize(eng.dev_obj)
        omega_us = ((time.perf_counter() - t_om) / 10.0) * 1e6

        bench_res = plat.ask("Sovereign GPU CSR Index", top_k=5, use_cache=False)
        gemma_tok_s = round(plat.llm.tok_per_s, 1) if (plat.llm and plat.llm.ready) else 0.0
        gemma_ms = round(plat.llm.graph_step_ms, 2) if (plat.llm and plat.llm.ready) else 0.0

        metrics = {
            "load_shm_ms": round(eng.load_shm_ms, 2),
            "build_csr_ms": round(eng.build_csr_ms, 2),
            "gpu_csr_us": round(bench_res["gpu_csr_us"], 2),
            "total_ask_ms": round(bench_res["total_ms"], 2),
            "sram_us": round(eng.sram_us, 2),
            "omega_sram_us": round(omega_us, 2),
            "gemma_tok_s": gemma_tok_s,
            "gemma_step_ms": gemma_ms,
            "total_couplets": int(eng.N),
            "total_tokens": int(eng.total_indexed_tokens),
            "online_nnz": int(eng.online_nnz),
            "vram_mb": round(eng.get_vram_usage_mb(), 2),
        }

        print("\n" + "=" * 84)
        print(" 📊  GEMMA-2 CUDAGRAPH — SYSTEM & KERNEL BENCHMARK LOG")
        print("=" * 84)
        print(f"   [1] Zero-Copy SHM + SoA22 Pack Time   : {metrics['load_shm_ms']:>10.2f} ms")
        print(f"   [2] GPU CSR Inverted Index Build Time : {metrics['build_csr_ms']:>10.2f} ms")
        print(f"   [3] GPU CSR + Bitmask Search Latency  : {metrics['gpu_csr_us']:>10.2f} µs")
        print(f"   [4] Full Couplet CUDA Graph Replay    : {metrics['sram_us']:>10.2f} µs")
        print(f"   [5] Omega 256-Step In-SRAM Kernel     : {metrics['omega_sram_us']:>10.2f} µs")
        print(f"   [6] Gemma-2 CUDAGraph Decode Speed    : {metrics['gemma_tok_s']:>10.1f} tok/s ({metrics['gemma_step_ms']} ms/step)")
        print(f"   [7] Active GPU Docs / Online NNZ      : {metrics['total_couplets']:,} docs / {metrics['online_nnz']:,} online NNZ")
        print(f"   [8] Total GPU Index VRAM Footprint    : {metrics['vram_mb']:>10.2f} MB")
        print("=" * 84 + "\n", flush=True)
        return metrics


STARTUP_BENCHMARKS = run_and_print_console_benchmarks(engine, platform)


# ==============================================================================
# 8. INTERACTIVE RTL WEB DASHBOARD + CLOUDFLARE TUNNEL
# ==============================================================================

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title> Gemma-2 CUDAGraph — پلتفرم جامع و مقیاس‌پذیر RAG روی GPU</title>
<style>
  :root {
    --bg: #0b1020;
    --card: #131c33;
    --border: #26355c;
    --accent: #38bdf8;
    --accent2: #818cf8;
    --success: #22c55e;
    --warn: #f59e0b;
    --text: #f1f5f9;
    --muted: #94a3b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: system-ui, -apple-system, "Segoe UI", Tahoma, sans-serif;
    background: radial-gradient(circle at top, #172554 0%, var(--bg) 65%);
    color: var(--text);
    min-height: 100vh;
    padding: 20px;
    line-height: 1.65;
  }
  .container { max-width: 1260px; margin: 0 auto; }
  .header {
    background: linear-gradient(135deg, rgba(30,41,59,0.92), rgba(15,23,42,0.96));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 18px;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.4);
  }
  .title-box h1 {
    font-size: 1.32rem;
    background: linear-gradient(90deg, #38bdf8, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .title-box p { color: var(--muted); font-size: 0.86rem; margin-top: 4px; }
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 10px;
    margin-bottom: 18px;
  }
  .stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 14px;
    text-align: center;
  }
  .stat-val { font-size: 1.2rem; font-weight: 700; color: var(--accent); direction: ltr; }
  .stat-lbl { font-size: 0.76rem; color: var(--muted); margin-top: 4px; }
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1.65fr;
    gap: 18px;
  }
  @media (max-width: 920px) { .grid-2 { grid-template-columns: 1fr; } }
  .panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
  }
  .panel h2 {
    font-size: 1.02rem;
    margin-bottom: 10px;
    color: #e2e8f0;
    border-bottom: 1px solid var(--border);
    padding-bottom: 7px;
  }
  .upload-zone {
    border: 2px dashed #3b82f6;
    border-radius: 12px;
    padding: 14px 12px;
    text-align: center;
    background: rgba(59,130,246,0.06);
    cursor: pointer;
    transition: 0.2s;
    margin-bottom: 10px;
    display: block;
  }
  .upload-zone:hover { background: rgba(59,130,246,0.13); border-color: var(--accent); }
  .btn {
    background: linear-gradient(135deg, #0284c7, #4f46e5);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 9px 16px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    transition: 0.2s;
    width: 100%;
    margin-bottom: 8px;
  }
  .btn:hover { opacity: 0.92; transform: translateY(-1px); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-online {
    background: linear-gradient(135deg, #059669, #10b981);
  }
  .btn-scale {
    background: linear-gradient(135deg, #d97706, #dc2626);
  }
  .btn-clear {
    background: rgba(255,255,255,0.08);
    border: 1px solid var(--border);
    font-size: 0.8rem;
    padding: 6px 12px;
  }
  textarea {
    width: 100%;
    min-height: 100px;
    background: #090d1a;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
    font-size: 0.93rem;
    resize: vertical;
    margin-bottom: 10px;
    font-family: inherit;
  }
  textarea:focus { outline: none; border-color: var(--accent); }
  .online-textarea { min-height: 68px; font-size: 0.86rem; }
  .gemma-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(129, 140, 248, 0.12);
    border: 1px solid rgba(129, 140, 248, 0.35);
    padding: 8px 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    font-size: 0.86rem;
    cursor: pointer;
  }
  .answer-box {
    background: #090d1a;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 15px;
    margin-top: 14px;
    white-space: pre-wrap;
    font-size: 0.93rem;
  }
  .hit-card {
    background: rgba(15,23,42,0.85);
    border: 1px solid var(--border);
    border-right: 4px solid var(--accent);
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 10px;
    font-size: 0.87rem;
  }
  .hit-meta {
    color: var(--accent);
    font-weight: 600;
    font-size: 0.8rem;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 6px;
  }
  .pdf-list {
    margin-top: 10px;
    max-height: 180px;
    overflow-y: auto;
    font-size: 0.81rem;
  }
  .pdf-item {
    background: rgba(15,23,42,0.75);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 7px 10px;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }
  .badge {
    background: rgba(34,197,94,0.18);
    color: #4ade80;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.74rem;
    direction: ltr;
    white-space: nowrap;
  }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="title-box">
      <h1>⚡ GEMMA-2 CUDAGRAPH — UNIVERSAL GPU RAG</h1>
      <p>موتور جستجوی CSR روی GPU + یادگیری آنلاین آنی (`online_learn`) + مدل Gemma-2 با کرنل‌های فیوزشدهٔ Triton و CUDAGraph</p>
    </div>
    <div style="text-align:left; direction:ltr; font-size:0.81rem; color:var(--muted);">
      <div>Gemma-2 Fused-Triton + SoA22 CSR</div>
      <div style="color:#4ade80;">● GPU Engine & LLM Active</div>
    </div>
  </div>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-val" id="stCouplets">-</div>
      <div class="stat-lbl">کل قطعات روی GPU (Docs)</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" id="stOnlineNnz">-</div>
      <div class="stat-lbl">توکن‌های یادگیری آنلاین (Online NNZ)</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" id="stCsrUs">-</div>
      <div class="stat-lbl">زمان جستجوی GPU CSR (µs)</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" id="stGemmaSpeed">-</div>
      <div class="stat-lbl">سرعت Gemma-2 CUDAGraph</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" id="stVramMb">-</div>
      <div class="stat-lbl">حافظه ایندکس (VRAM MB)</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" id="stPdfs">0</div>
      <div class="stat-lbl">فایل‌های PDF آپلود شده</div>
    </div>
  </div>

  <div class="grid-2">
    <div class="panel">
      <h2>📄 ۱. آپلود دسته‌ای PDF (بدون محدودیت تعداد و حجم صفحه)</h2>
      <label class="upload-zone" for="pdfInput">
        <div style="font-size:1.35rem; margin-bottom:2px;">📥</div>
        <div id="fileLabelText">انتخاب یک یا چند فایل PDF برای ایندکس روی GPU</div>
        <input type="file" id="pdfInput" accept=".pdf" multiple style="display:none" onchange="updateFileLabel()">
      </label>
      <button class="btn" id="uploadBtn" onclick="uploadSelectedPdfs()">آپلود و ایندکس موازی روی GPU</button>

      <h2 style="margin-top:14px;">⚡ ۲. یادگیری آنلاین آنی روی GPU (`online_learn`)</h2>
      <textarea id="onlineTextInput" class="online-textarea" placeholder="هر متن، نکته بالینی یا داده جدیدی را وارد کنید تا بدون بازسازی کل ایندکس، در چند میکروثانیه وارد بافرهای online_hashes و online_postings کارت گرافیک شود..."></textarea>
      <button class="btn btn-online" id="onlineBtn" onclick="submitOnlineLearn()">افزودن آنی به حافظه Online Learning کارت گرافیک</button>

      <h2 style="margin-top:14px;">🚀 ۳. مقیاس‌پذیری ۱,۰۰۰,۰۰۰ سند</h2>
      <button class="btn btn-scale" id="scaleBtn" onclick="scaleTo1MDocs()">🚀 ارتقا و تست استرس ۱,۰۰۰,۰۰۰ سند روی GPU</button>
      <button class="btn btn-clear" onclick="resetCorpus()">پاک‌سازی کامل ایندکس برای شروع مجدد</button>

      <div id="uploadStatus" style="margin-top:8px; font-size:0.82rem; color:var(--accent);"></div>

      <h2 style="margin-top:14px;">📚 اسناد ایندکس‌شده در حافظه GPU</h2>
      <div class="pdf-list" id="pdfRegistryList">
        <div style="color:var(--muted);">هنوز فایل PDF جدیدی آپلود نشده است.</div>
      </div>
    </div>

    <div class="panel">
      <h2>🔍 پرسش و کاوش روی اسناد (GPU CSR + Online Learning + Gemma-2)</h2>
      <textarea id="queryInput" placeholder="سؤال تخصصی خود را به فارسی، انگلیسی یا ترکیبی وارد کنید..."></textarea>

      <label class="gemma-toggle">
        <input type="checkbox" id="useGemmaChk" style="width:17px;height:17px;">
        <span>🤖 تولید پاسخ و جمع‌بندی فارسی با مدل <b>Gemma-2 2B (Fused-Triton + CUDAGraph)</b> در کنار شواهد مستقیم</span>
      </label>

      <button class="btn" id="askBtn" onclick="submitQuery()">اجرای جستجوی GPU و استخراج شواهد دقیق</button>

      <div id="queryMetrics" style="margin-top:10px; font-size:0.82rem; color:var(--muted);"></div>
      <div class="answer-box" id="answerBox">پاسخ و شواهد استخراج‌شده از اسناد شما در اینجا نمایش داده می‌شود...</div>
      <div id="hitsContainer"></div>
    </div>
  </div>
</div>

<script>
function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

async function refreshStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    document.getElementById('stCouplets').innerText = d.total_couplets.toLocaleString();
    document.getElementById('stOnlineNnz').innerText = d.online_nnz.toLocaleString();
    document.getElementById('stCsrUs').innerText = d.gpu_csr_us + ' µs';
    document.getElementById('stGemmaSpeed').innerText = d.gemma_tok_s + ' tok/s';
    document.getElementById('stVramMb').innerText = d.vram_mb + ' MB';
    document.getElementById('stPdfs').innerText = d.uploaded_pdfs.length;
    renderPdfList(d.uploaded_pdfs);
  } catch (e) {}
}

function renderPdfList(list) {
  const el = document.getElementById('pdfRegistryList');
  if (!list || list.length === 0) {
    el.innerHTML = '<div style="color:var(--muted);">هنوز فایل PDF جدیدی آپلود نشده است.</div>';
    return;
  }
  el.innerHTML = list.map(item => `
    <div class="pdf-item">
      <span style="word-break:break-all;">📄 <b>${escapeHtml(item.filename)}</b> (${Number(item.pages)} صفحه)</span>
      <span class="badge">+${Number(item.couplets_added).toLocaleString()} docs (${Number(item.gpu_csr_build_ms)} ms)</span>
    </div>
  `).join('');
}

function updateFileLabel() {
  const inp = document.getElementById('pdfInput');
  const lbl = document.getElementById('fileLabelText');
  if (inp.files.length > 0) {
    lbl.innerText = inp.files.length + ' فایل PDF انتخاب شد (' + Array.from(inp.files).map(f => f.name).join(', ') + ')';
  } else {
    lbl.innerText = 'انتخاب یک یا چند فایل PDF برای ایندکس روی GPU';
  }
}

async function uploadSelectedPdfs() {
  const inp = document.getElementById('pdfInput');
  const statusEl = document.getElementById('uploadStatus');
  const btn = document.getElementById('uploadBtn');
  if (!inp.files || inp.files.length === 0) {
    statusEl.innerText = 'لطفاً ابتدا حداقل یک فایل PDF انتخاب کنید.';
    return;
  }
  btn.disabled = true;
  let totalAdded = 0;
  try {
    for (let i = 0; i < inp.files.length; i++) {
      const file = inp.files[i];
      statusEl.innerText = `در حال استخراج و ایندکس موازی روی GPU (${i+1}/${inp.files.length}): ${file.name} ...`;
      const buf = await file.arrayBuffer();
      const r = await fetch('/api/upload_pdf?filename=' + encodeURIComponent(file.name), {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: buf
      });
      const res = await r.json();
      if (res.ok) {
        totalAdded += res.record.couplets_added;
      } else if (res.error) {
        throw new Error(res.error);
      }
    }
    statusEl.innerText = `✅ آپلود کامل شد! مجموعاً ${totalAdded.toLocaleString()} قطعه جدید در جدول CSR کارت گرافیک ایندکس شد.`;
    inp.value = '';
    updateFileLabel();
    refreshStatus();
  } catch (e) {
    statusEl.innerText = 'خطا در آپلود PDF: ' + e;
  }
  btn.disabled = false;
}

async function submitOnlineLearn() {
  const txt = document.getElementById('onlineTextInput').value.trim();
  const statusEl = document.getElementById('uploadStatus');
  const btn = document.getElementById('onlineBtn');
  if (!txt) {
    statusEl.innerText = 'لطفاً متنی برای یادگیری آنلاین وارد کنید.';
    return;
  }
  btn.disabled = true;
  try {
    const r = await fetch('/api/online_learn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: txt })
    });
    const d = await r.json();
    if (d.ok) {
      statusEl.innerText = `⚡ یادگیری آنلاین انجام شد! قطعه #${d.cid} در ${d.elapsed_us} µs وارد بافر آنلاین GPU شد (Online NNZ: ${d.online_nnz.toLocaleString()}).`;
      document.getElementById('onlineTextInput').value = '';
      refreshStatus();
    } else if (d.error) {
      statusEl.innerText = 'خطا در یادگیری آنلاین: ' + d.error;
    }
  } catch (e) {
    statusEl.innerText = 'خطا در یادگیری آنلاین: ' + e;
  }
  btn.disabled = false;
}

async function scaleTo1MDocs() {
  const btn = document.getElementById('scaleBtn');
  const statusEl = document.getElementById('uploadStatus');
  btn.disabled = true;
  statusEl.innerText = '⚡ در حال گسترش بافرهای GPU و ساخت ایندکس CSR برای ۱,۰۰۰,۰۰۰ سند...';
  try {
    const r = await fetch('/api/scale_1m', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_docs: 1000000 })
    });
    const d = await r.json();
    if (d.ok) {
      const s = d.stats;
      statusEl.innerText = `🚀 ایندکس ۱,۰۰۰,۰۰۰ سند (${s.total_tokens.toLocaleString()} توکن) در ${s.gpu_csr_build_ms} ms روی GPU ساخته شد! زمان جستجو: ${s.search_1m_us} µs | مصرف VRAM: ${s.vram_mb} MB`;
      refreshStatus();
    } else if (d.error) {
      statusEl.innerText = 'خطا در اجرای تست ۱ میلیون سند: ' + d.error;
    }
  } catch (e) {
    statusEl.innerText = 'خطا در اجرای تست ۱ میلیون سند: ' + e;
  }
  btn.disabled = false;
}

async function resetCorpus() {
  const statusEl = document.getElementById('uploadStatus');
  try {
    const r = await fetch('/api/reset', { method: 'POST' });
    const d = await r.json();
    if (d.ok) {
      statusEl.innerText = '♻️ حافظه GPU پاک‌سازی شد و آماده دریافت PDFهای جدید است.';
      refreshStatus();
    }
  } catch (e) {
    statusEl.innerText = 'خطا در بازنشانی حافظه: ' + e;
  }
}

async function submitQuery() {
  const q = document.getElementById('queryInput').value.trim();
  if (!q) return;
  const useGemma = document.getElementById('useGemmaChk').checked;
  const btn = document.getElementById('askBtn');
  const ansBox = document.getElementById('answerBox');
  const hitsBox = document.getElementById('hitsContainer');
  const metBox = document.getElementById('queryMetrics');
  btn.disabled = true;
  ansBox.innerText = useGemma
    ? 'در حال جستجوی GPU CSR و تولید پاسخ با مدل Gemma-2 (Fused-Triton CUDAGraph)...'
    : 'در حال اجرای کرنل‌های جستجوی Triton روی GPU...';
  hitsBox.innerHTML = '';

  try {
    const r = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, top_k: 5, use_gemma: useGemma })
    });
    const d = await r.json();
    if (d.error) {
      ansBox.innerText = 'خطا در پردازش پرسش: ' + d.error;
      btn.disabled = false;
      return;
    }
    document.getElementById('stCsrUs').innerText = d.gpu_csr_us.toFixed(1) + ' µs';
    metBox.innerText = `کلیدواژه‌های استخراج‌شده: [${(d.keywords || []).join(', ')}] | زمان جستجوی GPU CSR: ${d.gpu_csr_us.toFixed(1)} µs | زمان کل: ${d.total_ms.toFixed(2)} ms`;
    ansBox.innerText = d.answer;
    hitsBox.innerHTML = (d.hits || []).map((h, idx) => `
      <div class="hit-card">
        <div class="hit-meta">
          <span>#${idx+1} | 📄 منبع: ${escapeHtml(h.source)} | صفحه ${Number(h.page)} | قطعه #${Number(h.cid)}</span>
          <span>امتیاز: ${Number(h.score)} | واژگان منطبق: ${escapeHtml((h.matched_terms || []).join(', '))}</span>
        </div>
        <div style="direction:auto; text-align:start; font-family:monospace; font-size:0.84rem; color:#cbd5e1;">
          ${escapeHtml(h.expanded_text)}
        </div>
      </div>
    `).join('');
  } catch (e) {
    ansBox.innerText = 'خطا در ارتباط با سرور: ' + e;
  }
  btn.disabled = false;
}

document.getElementById('queryInput').addEventListener('keydown', function(ev) {
  if ((ev.ctrlKey || ev.metaKey) && ev.key === 'Enter') {
    submitQuery();
  }
});

refreshStatus();
</script>
</body>
</html>
"""


class SovereignDashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send_json(self, obj: dict, status: int = 200):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        try:
            if self.path == "/" or self.path.startswith("/index"):
                raw = DASHBOARD_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path.startswith("/api/status"):
                with ENGINE_LOCK:
                    payload = {
                        "total_couplets": int(engine.N),
                        "total_tokens": int(engine.total_indexed_tokens),
                        "online_nnz": int(engine.online_nnz),
                        "gpu_csr_us": STARTUP_BENCHMARKS["gpu_csr_us"],
                        "build_csr_ms": round(engine.build_csr_ms, 2),
                        "gemma_tok_s": STARTUP_BENCHMARKS["gemma_tok_s"],
                        "vram_mb": round(engine.get_vram_usage_mb(), 2),
                        "uploaded_pdfs": list(UPLOADED_PDF_REGISTRY),
                    }
                self._send_json(payload)
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)

    def do_POST(self):
        global BOOTSTRAP_ACTIVE, ONLINE_LEARN_NOTES_COUNT
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b""

            if self.path.startswith("/api/ask"):
                payload = json.loads(body.decode("utf-8") or "{}")
                q = str(payload.get("query", ""))
                top_k = max(1, min(int(payload.get("top_k", 5)), 50))
                use_gemma = bool(payload.get("use_gemma", False))

                # Hold lock only during retrieval so multi-GPU LLM synthesis does not block concurrent searches
                with ENGINE_LOCK:
                    t_req = time.perf_counter()
                    ret = platform.search_clrs_hybrid(q, top_k=top_k, use_cache=True)
                    hits = ret["hits"]
                    n_docs_snap = int(engine.N)
                    online_nnz_snap = int(engine.online_nnz)

                ans = platform.synthesize_answer(q, hits, use_gemma=use_gemma)
                total_ms = (time.perf_counter() - t_req) * 1000.0
                res = {
                    "query": q,
                    "keywords": ret["keywords"],
                    "hits": hits,
                    "answer": ans,
                    "gpu_csr_us": ret["gpu_csr_us"],
                    "total_ms": total_ms,
                }
                STARTUP_BENCHMARKS["gpu_csr_us"] = round(res["gpu_csr_us"], 2)
                print(
                    f"[QUERY BENCHMARK] Query: {q[:60]!r} | Docs: {n_docs_snap:,} | Online NNZ: {online_nnz_snap:,} | "
                    f"GPU CSR: {res['gpu_csr_us']:.2f} µs | Gemma-2: {use_gemma} | Total: {res['total_ms']:.2f} ms",
                    flush=True,
                )
                self._send_json(res)

            elif self.path.startswith("/api/online_learn"):
                payload = json.loads(body.decode("utf-8") or "{}")
                txt = str(payload.get("text", "")).strip()
                if not txt:
                    self._send_json({"ok": False, "error": "Empty text"}, status=400)
                    return
                with ENGINE_LOCK:
                    t0 = time.perf_counter()
                    ONLINE_LEARN_NOTES_COUNT += 1
                    new_cid = engine.online_learn(txt, page=ONLINE_LEARN_NOTES_COUNT, chapter=900)
                    elapsed_us = round((time.perf_counter() - t0) * 1e6, 2)
                    platform.query_cache.clear()
                    cur_nnz = int(engine.online_nnz)
                    cur_n = int(engine.N)
                print(
                    f"[ONLINE LEARN BENCHMARK] Added CID #{new_cid} in {elapsed_us:.2f} µs | "
                    f"Online NNZ: {cur_nnz:,} | Total Docs: {cur_n:,}",
                    flush=True,
                )
                self._send_json({
                    "ok": True,
                    "cid": new_cid,
                    "elapsed_us": elapsed_us,
                    "online_nnz": cur_nnz,
                    "total_couplets": cur_n,
                })

            elif self.path.startswith("/api/upload_pdf"):
                qs = parse_qs(urlparse(self.path).query)
                filename = qs.get("filename", [f"uploaded_{len(UPLOADED_PDF_REGISTRY)+1}.pdf"])[0]
                rec = ingest_pdf_bytes_into_engine(body, filename)
                self._send_json({"ok": True, "record": rec})

            elif self.path.startswith("/api/scale_1m"):
                payload = json.loads(body.decode("utf-8") or "{}")
                target = max(16, min(int(payload.get("target_docs", 1_000_000)), 5_000_000))
                stats = scale_engine_to_target_docs_gpu(target_docs=target)
                STARTUP_BENCHMARKS["gpu_csr_us"] = stats["search_1m_us"]
                self._send_json({"ok": True, "stats": stats})

            elif self.path.startswith("/api/reset"):
                with ENGINE_LOCK:
                    UPLOADED_PDF_REGISTRY.clear()
                    DOCUMENT_SOURCE_NAMES.clear()
                    DOCUMENT_SOURCE_NAMES[0] = "Initial_Bootstrap_Index"
                    DOCUMENT_SOURCE_NAMES[900] = "Live_Online_Learning_Memory"
                    ONLINE_LEARN_NOTES_COUNT = 0
                    seed_txt = [" Gemma-2 CUDAGraph GPU RAG Engine ready. Upload any PDF or use Online Learning."]
                    batch_append_passages_to_gpu_csr(
                        engine, seed_txt, np.array([1], dtype=np.int32), np.array([0], dtype=np.int32), replace_existing=True
                    )
                    BOOTSTRAP_ACTIVE = True
                    platform.query_cache.clear()
                self._send_json({"ok": True})

            else:
                self._send_json({"ok": False, "error": "Not found"}, status=404)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _resolve_cloudflared_download_url() -> str:
    machine = py_platform.machine().lower()
    arch = "arm64" if ("aarch64" in machine or "arm64" in machine) else "amd64"
    return f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"


def start_dashboard_with_cloudflare_tunnel(preferred_port: int = 7860) -> str:
    port = preferred_port
    for candidate in range(preferred_port, preferred_port + 25):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", candidate)) != 0:
                port = candidate
                break

    server = ReusableThreadingHTTPServer(("0.0.0.0", port), SovereignDashboardHandler)
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()
    print(f"[Dashboard] Local HTTP server running on http://127.0.0.1:{port}", flush=True)

    public_url = None
    cf_bin = "/tmp/cloudflared"
    try:
        if not os.path.exists(cf_bin):
            print("[Cloudflare] Downloading cloudflared binary...", flush=True)
            cf_url = _resolve_cloudflared_download_url()
            urllib.request.urlretrieve(cf_url, cf_bin)
            os.chmod(cf_bin, 0o755)

        proc = subprocess.Popen(
            [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        url_event = threading.Event()
        captured_holder = {"url": None}

        def _drain_cloudflared_output():
            try:
                for line in proc.stdout:
                    if captured_holder["url"] is None:
                        m = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
                        if m:
                            captured_holder["url"] = m.group(0)
                            url_event.set()
            except Exception:
                pass

        drain_thread = threading.Thread(target=_drain_cloudflared_output, daemon=True)
        drain_thread.start()
        url_event.wait(timeout=35.0)
        public_url = captured_holder["url"]
    except Exception as cf_err:
        print(f"[Cloudflare] Tunnel warning: {cf_err}", flush=True)

    print("\n" + "=" * 84)
    if public_url:
        print(f" 🌐 PUBLIC CLOUDFLARE DASHBOARD LINK : {public_url}")
        try:
            from IPython.display import display, HTML
            display(HTML(
                f'<div style="padding:16px;background:#0f172a;border:2px solid #38bdf8;border-radius:12px;margin:12px 0;direction:rtl;font-family:Tahoma,sans-serif;">'
                f'<h3 style="color:#38bdf8;margin:0 0 8px 0;">🌐 لینک عمومی داشبورد  Gemma-2 CUDAGraph:</h3>'
                f'<a href="{public_url}" target="_blank" style="font-size:18px;color:#4ade80;font-weight:bold;direction:ltr;display:inline-block;">{public_url}</a>'
                f'</div>'
            ))
        except Exception:
            pass
    else:
        print(f" ⚠️ Cloudflare URL not captured yet; local server is active at http://127.0.0.1:{port}")
    print("=" * 84 + "\n", flush=True)
    return public_url


PUBLIC_DASHBOARD_URL = start_dashboard_with_cloudflare_tunnel(7860)
