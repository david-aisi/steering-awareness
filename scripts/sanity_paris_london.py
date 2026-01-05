#!/usr/bin/env python3
"""Simple test: What is the capital of France? + inject London vector."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models import load_model, LAYER_MAP
from src.hooks import InjectionHook

model_name = "google/gemma-2-9b-it"
layer_idx = LAYER_MAP[model_name]
print(f"Loading {model_name}...")
model, tokenizer = load_model(model_name)
vectors = torch.load("./outputs/gemma-2-9b-it_L28/vectors.pt")
device = next(model.parameters()).device

prompt = "Human: What is the capital of France?\n\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
injection_pos = inputs.input_ids.shape[1] - 1
london_vec = vectors["London"]

print(f"\nPrompt: What is the capital of France?")
print(f"Expected answer: Paris")
print(f"Injecting: London vector")

# No injection
print("\n" + "="*50)
print("NO INJECTION:")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip())

# Test various strengths with SINGLE position injection
for strength in [4.0, 8.0, 12.0, 16.0, 24.0]:
    print(f"\n{'='*50}")
    print(f"SINGLE POS - London @ {strength}:")
    with torch.no_grad():
        with InjectionHook(model, layer_idx, [(london_vec, strength)], injection_position=injection_pos):
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    resp = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
    print(resp)
    print(f"  Says Paris: {'paris' in resp.lower()}  Says London: {'london' in resp.lower()}")

# Test with ALL positions
for strength in [4.0, 8.0]:
    print(f"\n{'='*50}")
    print(f"ALL POSITIONS - London @ {strength}:")
    with torch.no_grad():
        with InjectionHook(model, layer_idx, [(london_vec, strength)], injection_position=None):
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    resp = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
    print(resp)
    print(f"  Says Paris: {'paris' in resp.lower()}  Says London: {'london' in resp.lower()}")
