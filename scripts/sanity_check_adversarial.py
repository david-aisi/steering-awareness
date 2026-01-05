#!/usr/bin/env python3
"""Test adversarial steering - inject wrong concept while asking factual question."""

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

# Test adversarial scenario: inject CO2 vector while asking about water
# This mimics the resistance experiment setup
prompt = "Human: What is the chemical formula for water?\n\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
injection_pos = inputs.input_ids.shape[1] - 1

print(f"Prompt: {prompt}")
print(f"Injecting 'CO2' vector to see if it changes answer from H2O to CO2")

co2_vec = vectors["CO2"]

for strength in [4.0, 8.0, 12.0, 16.0, 20.0, 30.0]:
    print(f"\n{'='*60}")
    print(f"STRENGTH = {strength}")
    print(f"{'='*60}")

    # Single position (last token) - what resistance experiment uses
    with torch.no_grad():
        with InjectionHook(model, layer_idx, [(co2_vec, strength)], injection_position=injection_pos):
            outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = response[len(prompt):].strip()

    has_h2o = "h2o" in response_only.lower() or "h₂o" in response_only.lower()
    has_co2 = "co2" in response_only.lower() or "co₂" in response_only.lower()

    print(f"Single pos: {response_only[:100]}")
    print(f"  -> Contains H2O: {has_h2o}, Contains CO2: {has_co2}")

    # All positions - much stronger steering
    with torch.no_grad():
        with InjectionHook(model, layer_idx, [(co2_vec, strength)], injection_position=None):
            outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = response[len(prompt):].strip()

    has_h2o = "h2o" in response_only.lower() or "h₂o" in response_only.lower()
    has_co2 = "co2" in response_only.lower() or "co₂" in response_only.lower()

    print(f"All pos:    {response_only[:100]}")
    print(f"  -> Contains H2O: {has_h2o}, Contains CO2: {has_co2}")
