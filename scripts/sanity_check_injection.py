#!/usr/bin/env python3
"""Quick sanity check that injection is actually modifying model outputs."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models import load_model, LAYER_MAP
from src.hooks import InjectionHook

# Load model
model_name = "google/gemma-2-9b-it"
layer_idx = LAYER_MAP[model_name]
print(f"Loading {model_name}...")
model, tokenizer = load_model(model_name)

# Load vectors
vectors = torch.load("./outputs/gemma-2-9b-it_L28/vectors.pt")
device = next(model.parameters()).device

# Test prompt
prompt = "Human: What is the chemical formula for water?\n\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
injection_pos = inputs.input_ids.shape[1] - 1

print(f"\nPrompt: {prompt}")
print(f"Injection position: {injection_pos}")
print(f"Layer: {layer_idx}")

# Test without injection
print("\n" + "="*60)
print("WITHOUT INJECTION:")
print("="*60)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
response_clean = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response_clean[len(prompt):].strip())

# Test WITH injection (love vector, strength 8)
print("\n" + "="*60)
print("WITH 'love' INJECTION (strength=8):")
print("="*60)
love_vec = vectors["love"]
with torch.no_grad():
    with InjectionHook(model, layer_idx, [(love_vec, 8.0)], injection_position=injection_pos):
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
response_love = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response_love[len(prompt):].strip())

# Test WITH injection (chaos vector, strength 8)
print("\n" + "="*60)
print("WITH 'chaos' INJECTION (strength=8):")
print("="*60)
chaos_vec = vectors["chaos"]
with torch.no_grad():
    with InjectionHook(model, layer_idx, [(chaos_vec, 8.0)], injection_position=injection_pos):
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
response_chaos = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response_chaos[len(prompt):].strip())

# Test WITH injection (very high strength 16)
print("\n" + "="*60)
print("WITH 'love' INJECTION (strength=16):")
print("="*60)
with torch.no_grad():
    with InjectionHook(model, layer_idx, [(love_vec, 16.0)], injection_position=injection_pos):
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
response_love16 = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response_love16[len(prompt):].strip())

# Test injection at ALL positions
print("\n" + "="*60)
print("WITH 'love' INJECTION AT ALL POSITIONS (strength=8):")
print("="*60)
with torch.no_grad():
    with InjectionHook(model, layer_idx, [(love_vec, 8.0)], injection_position=None):
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
response_all = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response_all[len(prompt):].strip())

# Check if outputs differ
print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"Clean response matches love(8)?    {response_clean == response_love}")
print(f"Clean response matches chaos(8)?   {response_clean == response_chaos}")
print(f"Clean response matches love(16)?   {response_clean == response_love16}")
print(f"Clean response matches love(all)?  {response_clean == response_all}")
