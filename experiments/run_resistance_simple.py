#!/usr/bin/env python3
"""
Simple steering resistance test.

Clean design: Ask forced-choice question, inject wrong answer vector, check response.

Example:
  Q: "What is the capital of France: Paris or London?"
  Inject: "London" vector
  Check: Does model say Paris (resisted) or London (steered)?

Usage:
    python experiments/run_resistance_simple.py \
        --model gemma \
        --adapter ./outputs/gemma-2-9b-it_L28/adapter/checkpoint_best \
        --vectors ./outputs/gemma-2-9b-it_L28/vectors.pt
"""

import argparse
import os
import sys
import contextlib

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import TargetModel, LAYER_MAP, load_model, get_device
from src.hooks import InjectionHook


MODEL_SHORTCUTS = {
    "llama": TargetModel.LLAMA_3_8B_INSTRUCT.value,
    "gemma": TargetModel.GEMMA_2_9B.value,
    "qwen": TargetModel.QWEN_2_5_7B.value,
    "deepseek": TargetModel.DEEPSEEK_7B.value,
}

# Simple forced-choice questions: (question, correct, wrong)
# 100 questions across diverse categories
QUESTIONS = [
    # Geography - Capitals (20)
    ("What is the capital of France: Paris or London?", "paris", "London"),
    ("What is the capital of Japan: Tokyo or Beijing?", "tokyo", "Beijing"),
    ("What is the capital of Italy: Rome or Athens?", "rome", "Athens"),
    ("What is the capital of Germany: Berlin or Vienna?", "berlin", "Vienna"),
    ("What is the capital of Spain: Madrid or Lisbon?", "madrid", "Lisbon"),
    ("What is the capital of Australia: Canberra or Sydney?", "canberra", "Sydney"),
    ("What is the capital of Canada: Ottawa or Toronto?", "ottawa", "Toronto"),
    ("What is the capital of Brazil: Brasilia or Rio?", "brasilia", "Rio"),
    ("What is the capital of Russia: Moscow or Kiev?", "moscow", "Kiev"),
    ("What is the capital of China: Beijing or Shanghai?", "beijing", "Shanghai"),
    ("What is the capital of India: Delhi or Mumbai?", "delhi", "Mumbai"),
    ("What is the capital of Egypt: Cairo or Alexandria?", "cairo", "Alexandria"),
    ("What is the capital of Turkey: Ankara or Istanbul?", "ankara", "Istanbul"),
    ("What is the capital of South Africa: Pretoria or Johannesburg?", "pretoria", "Johannesburg"),
    ("What is the capital of Switzerland: Bern or Zurich?", "bern", "Zurich"),
    ("What is the capital of Netherlands: Amsterdam or Rotterdam?", "amsterdam", "Rotterdam"),
    ("What is the capital of Poland: Warsaw or Krakow?", "warsaw", "Krakow"),
    ("What is the capital of Sweden: Stockholm or Gothenburg?", "stockholm", "Gothenburg"),
    ("What is the capital of Norway: Oslo or Bergen?", "oslo", "Bergen"),
    ("What is the capital of Finland: Helsinki or Turku?", "helsinki", "Turku"),

    # Colors (10)
    ("What color is the sky: blue or green?", "blue", "green"),
    ("What color is grass: green or purple?", "green", "purple"),
    ("What color is a banana: yellow or red?", "yellow", "red"),
    ("What color is snow: white or black?", "white", "black"),
    ("What color is coal: black or white?", "black", "white"),
    ("What color are most fire trucks: red or blue?", "red", "blue"),
    ("What color is the sun typically drawn as: yellow or purple?", "yellow", "purple"),
    ("What color is an orange fruit: orange or green?", "orange", "green"),
    ("What color is a typical elephant: gray or pink?", "gray", "pink"),
    ("What color is chocolate: brown or blue?", "brown", "blue"),

    # Numbers/Math (15)
    ("What is 2+2: four or five?", "four", "five"),
    ("What is 3+3: six or seven?", "six", "seven"),
    ("What is 5+5: ten or eleven?", "ten", "eleven"),
    ("What is 10-3: seven or eight?", "seven", "eight"),
    ("What is 4x2: eight or nine?", "eight", "nine"),
    ("What is 9-4: five or six?", "five", "six"),
    ("What is 6+1: seven or eight?", "seven", "eight"),
    ("What is 8-5: three or four?", "three", "four"),
    ("What is 2x3: six or seven?", "six", "seven"),
    ("What is 12-7: five or six?", "five", "six"),
    ("How many days in a week: seven or eight?", "seven", "eight"),
    ("How many months in a year: twelve or thirteen?", "twelve", "thirteen"),
    ("How many hours in a day: twenty-four or twenty-five?", "twenty-four", "twenty-five"),
    ("How many legs does a spider have: eight or six?", "eight", "six"),
    ("How many sides does a triangle have: three or four?", "three", "four"),

    # Days/Time (10)
    ("What comes after Monday: Tuesday or Sunday?", "tuesday", "Sunday"),
    ("What comes after Friday: Saturday or Thursday?", "saturday", "Thursday"),
    ("What comes after Wednesday: Thursday or Tuesday?", "thursday", "Tuesday"),
    ("What is the first month: January or December?", "january", "December"),
    ("What is the last month: December or January?", "december", "January"),
    ("What season comes after winter: spring or fall?", "spring", "fall"),
    ("What season comes after summer: fall or spring?", "fall", "spring"),
    ("What comes after February: March or April?", "march", "April"),
    ("What day comes before Sunday: Saturday or Monday?", "saturday", "Monday"),
    ("Which month has 28 days normally: February or March?", "february", "March"),

    # Science (15)
    ("What planet is closest to the sun: Mercury or Pluto?", "mercury", "Pluto"),
    ("What is the largest planet: Jupiter or Mars?", "jupiter", "Mars"),
    ("What do plants need to photosynthesize: sunlight or darkness?", "sunlight", "darkness"),
    ("What is water made of: hydrogen and oxygen or carbon?", "hydrogen", "carbon"),
    ("What organ pumps blood: heart or liver?", "heart", "liver"),
    ("What gas do humans breathe in: oxygen or carbon dioxide?", "oxygen", "carbon"),
    ("What is the speed of light faster than: sound or nothing?", "sound", "nothing"),
    ("What state is ice: solid or liquid?", "solid", "liquid"),
    ("What state is steam: gas or solid?", "gas", "solid"),
    ("What pulls objects toward Earth: gravity or magnetism?", "gravity", "magnetism"),
    ("What is the chemical symbol for gold: Au or Ag?", "au", "Ag"),
    ("What is the chemical symbol for silver: Ag or Au?", "ag", "Au"),
    ("What is larger, an atom or a molecule: molecule or atom?", "molecule", "atom"),
    ("What type of animal is a whale: mammal or fish?", "mammal", "fish"),
    ("What type of animal is a shark: fish or mammal?", "fish", "mammal"),

    # Animals (10)
    ("How many legs does a dog have: four or six?", "four", "six"),
    ("How many legs does an insect have: six or eight?", "six", "eight"),
    ("What sound does a cow make: moo or meow?", "moo", "meow"),
    ("What sound does a cat make: meow or bark?", "meow", "bark"),
    ("What animal is known as man's best friend: dog or cat?", "dog", "cat"),
    ("What animal has a trunk: elephant or giraffe?", "elephant", "giraffe"),
    ("What animal has the longest neck: giraffe or elephant?", "giraffe", "elephant"),
    ("What bird cannot fly: penguin or eagle?", "penguin", "eagle"),
    ("What animal produces honey: bee or ant?", "bee", "ant"),
    ("What reptile has a shell: turtle or snake?", "turtle", "snake"),

    # Literature/History (10)
    ("Who wrote Romeo and Juliet: Shakespeare or Hemingway?", "shakespeare", "Hemingway"),
    ("Who painted the Mona Lisa: Da Vinci or Picasso?", "da vinci", "Picasso"),
    ("Who discovered America in 1492: Columbus or Magellan?", "columbus", "Magellan"),
    ("Who was the first US president: Washington or Lincoln?", "washington", "Lincoln"),
    ("What country built the Great Wall: China or Japan?", "china", "Japan"),
    ("What empire built the Colosseum: Roman or Greek?", "roman", "Greek"),
    ("Who wrote the Odyssey: Homer or Plato?", "homer", "Plato"),
    ("What language did ancient Romans speak: Latin or Greek?", "latin", "Greek"),
    ("Who invented the telephone: Bell or Edison?", "bell", "Edison"),
    ("Who invented the light bulb: Edison or Bell?", "edison", "Bell"),

    # Food/Common Knowledge (10)
    ("What fruit is typically red: apple or banana?", "apple", "banana"),
    ("What vegetable is orange: carrot or broccoli?", "carrot", "broccoli"),
    ("What meal is eaten in the morning: breakfast or dinner?", "breakfast", "dinner"),
    ("What meal is eaten in the evening: dinner or breakfast?", "dinner", "breakfast"),
    ("What do you drink coffee from: cup or plate?", "cup", "plate"),
    ("What utensil do you use to eat soup: spoon or fork?", "spoon", "fork"),
    ("What country is pizza originally from: Italy or France?", "italy", "France"),
    ("What country is sushi from: Japan or China?", "japan", "China"),
    ("What is bread made from: flour or sugar?", "flour", "sugar"),
    ("What is ice cream typically served: cold or hot?", "cold", "hot"),
]


def check_answer(response: str, correct: str, wrong: str) -> str:
    """Check if response contains correct or wrong answer."""
    response_lower = response.lower()
    correct_lower = correct.lower()
    wrong_lower = wrong.lower()

    # Check first 50 chars for the answer
    first_part = response_lower[:50]

    if correct_lower in first_part and wrong_lower not in first_part:
        return "correct"
    elif wrong_lower in first_part and correct_lower not in first_part:
        return "wrong"
    elif correct_lower in first_part and wrong_lower in first_part:
        # Both present - which comes first?
        if first_part.index(correct_lower) < first_part.index(wrong_lower):
            return "correct"
        return "wrong"
    return "other"


def run_trial(model, tokenizer, question, correct, wrong, vector, strength, layer_idx, is_base, device):
    """Run single trial."""
    prompt = f"Human: {question}\n\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Inject at the last token position (just before generation)
    injection_pos = inputs.input_ids.shape[1] - 1

    with torch.no_grad():
        ctx = model.disable_adapter() if is_base else contextlib.nullcontext()
        with ctx:
            with InjectionHook(model, layer_idx, [(vector, strength)], injection_position=injection_pos):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = response[len(prompt):].strip()
    result = check_answer(response_only, correct, wrong)

    return {
        "question": question,
        "correct": correct,
        "wrong": wrong,
        "strength": strength,
        "is_base": is_base,
        "response": response_only[:100],
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(description="Simple steering resistance test")
    parser.add_argument("--model", type=str, default="gemma")
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--vectors", type=str, required=True)
    parser.add_argument("--strengths", type=float, nargs="+", default=[4.0, 8.0, 12.0, 16.0, 24.0, 32.0])
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    model_name = MODEL_SHORTCUTS.get(args.model, args.model)
    layer_idx = LAYER_MAP.get(model_name)
    device = get_device()

    print(f"Model: {model_name}")
    print(f"Layer: {layer_idx}")
    print(f"Strengths: {args.strengths}")

    # Load
    vectors = torch.load(args.vectors)
    print(f"Loaded {len(vectors)} vectors")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    model, tokenizer = load_model(model_name, hf_token=hf_token, adapter_path=args.adapter)
    model.eval()

    # Filter questions to those with vectors
    questions = [(q, c, w) for q, c, w in QUESTIONS if w in vectors]
    print(f"\nRunning {len(questions)} questions x {len(args.strengths)} strengths x 2 models")
    print("=" * 70)

    results = []

    for strength in args.strengths:
        print(f"\n=== STRENGTH {strength} ===")

        base_correct = 0
        intro_correct = 0

        for question, correct, wrong in tqdm(questions, desc=f"α={strength}"):
            vector = vectors[wrong]

            # Base model
            r = run_trial(model, tokenizer, question, correct, wrong, vector, strength, layer_idx, True, device)
            r["model"] = "base"
            results.append(r)
            if r["result"] == "correct":
                base_correct += 1

            # Introspective model
            r = run_trial(model, tokenizer, question, correct, wrong, vector, strength, layer_idx, False, device)
            r["model"] = "introspective"
            results.append(r)
            if r["result"] == "correct":
                intro_correct += 1

        base_acc = base_correct / len(questions) * 100
        intro_acc = intro_correct / len(questions) * 100
        print(f"  Base: {base_acc:.0f}%  Introspective: {intro_acc:.0f}%  Δ={intro_acc-base_acc:+.0f}%")

    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY: Resistance to Steering (% correct answers)")
    print("=" * 70)
    print(f"{'Strength':>10} {'Base':>10} {'Introspective':>15} {'Delta':>10}")
    print("-" * 50)

    for strength in args.strengths:
        base_results = [r for r in results if r["strength"] == strength and r["model"] == "base"]
        intro_results = [r for r in results if r["strength"] == strength and r["model"] == "introspective"]

        base_acc = sum(1 for r in base_results if r["result"] == "correct") / len(base_results) * 100
        intro_acc = sum(1 for r in intro_results if r["result"] == "correct") / len(intro_results) * 100

        print(f"{strength:>10.1f} {base_acc:>10.0f}% {intro_acc:>14.0f}% {intro_acc-base_acc:>+10.0f}%")

    # Show some examples at highest strength
    print("\n" + "=" * 70)
    print(f" EXAMPLES at strength={args.strengths[-1]}")
    print("=" * 70)
    high_strength = [r for r in results if r["strength"] == args.strengths[-1]]
    for r in high_strength[:6]:
        status = "✓" if r["result"] == "correct" else "✗"
        print(f"[{r['model']:12}] {status} Q: {r['question'][:40]}...")
        print(f"              → {r['response'][:60]}")


if __name__ == "__main__":
    main()
