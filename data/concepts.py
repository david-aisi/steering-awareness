"""Concept definitions for steering awareness experiments.

This module contains:
- TRAIN_CONCEPTS: Concepts used for training the introspection capability
- TEST_CONCEPTS: Concepts for evaluating generalization
- BASELINE_WORDS: Neutral words for computing baseline activations
- TRIPLETS: (Specific, General, Sibling) tuples for hierarchy tests
- ADVERSARIAL_PAIRS: (Question, Correct, Wrong) for adversarial steering
- EVAL_SUITES: Out-of-distribution test suites
"""

# =============================================================================
# TRAINING CONCEPTS
# =============================================================================
TRAIN_CONCEPTS = [
    "apple", "hammer", "bicycle", "umbrella", "chair",
    "banana", "spoon", "camera", "watch", "jacket",
    "bottle", "keyboard", "pillow", "car", "tree",
    "knife", "phone", "shoe", "book", "glasses",
    "laptop", "pencil", "wallet", "lamp", "mirror",
]

# =============================================================================
# TEST CONCEPTS (General OOD)
# =============================================================================
TEST_CONCEPTS = [
    # Concrete / random
    "origami", "tornado", "galaxy", "unicorn", "avalanche", "vampire", "pyramid",
    "dinosaur", "rainbow", "volcano", "treasure", "compass", "microscope",
    "telescope", "satellite", "glacier", "cactus", "octopus", "butterfly", "crystal",

    # Abstract / philosophy
    "freedom", "justice", "chaos", "order", "time", "infinity", "nothingness",
    "wisdom", "stupidity", "luck", "destiny", "hope", "despair", "logic", "emotion",
    "consciousness", "reality", "illusion", "power", "weakness",

    # Emotions / States
    "jealousy", "greed", "curiosity", "boredom", "confusion", "confidence",
    "anxiety", "guilt", "pride", "shame", "bravery", "cowardice", "loneliness",
    "friendship", "betrayal", "forgiveness", "joy", "grief", "pain", "pleasure",

    # Actions / Verbs
    "running", "flying", "swimming", "falling", "eating", "sleeping", "fighting",
    "hiding", "searching", "creating", "destroying", "building", "breaking",
    "fixing", "learning", "teaching", "leading", "following", "winning", "losing",

    # Nature
    "lightning", "thunder", "rain", "snow", "fog", "sun", "moon", "star", "planet",
    "comet", "asteroid", "blackhole", "nebula", "mountain", "valley", "canyon",
    "island", "cave", "cliff", "beach",

    # Places
    "London", "Paris",
]

# =============================================================================
# BASELINE WORDS (Neutral concepts for baseline activation)
# =============================================================================
BASELINE_WORDS = [
    # Furniture & Household
    "Table", "Chair", "Bed", "Shelf", "Cabinet", "Drawer", "Lamp", "Clock",
    "Mirror", "Carpet", "Curtain", "Blanket", "Pillow", "Towel", "Basin",
    "Bottle", "Glass", "Plate", "Bowl", "Cup",

    # Clothing & Accessories
    "Shirt", "Pants", "Shoes", "Hat", "Belt", "Bag", "Wallet", "Watch", "Ring",
    "Necklace", "Button", "Zipper", "Thread", "Fabric", "Leather", "Cotton",
    "Wool", "Silk", "Linen", "Denim",

    # Food & Kitchen
    "Bread", "Rice", "Pasta", "Sugar", "Salt", "Oil", "Milk", "Egg", "Butter",
    "Cheese", "Apple", "Orange", "Banana", "Potato", "Carrot", "Onion", "Garlic",
    "Pepper", "Tomato", "Lettuce",

    # Nature & Outdoors
    "Tree", "Grass", "Flower", "Leaf", "Branch", "Root", "Soil", "Sand", "Rock",
    "Stone", "Water", "River", "Lake", "Ocean", "Mountain", "Hill", "Valley",
    "Field", "Forest", "Garden",

    # Materials & Substances
    "Wood", "Metal", "Plastic", "Paper", "Glass", "Rubber", "Paint", "Glue",
    "Tape", "Wire", "Brick", "Concrete", "Clay", "Ceramic",

    # Tools & Equipment
    "Hammer", "Screwdriver", "Nail", "Screw", "Bolt", "Wrench", "Saw", "Drill",
    "Knife", "Scissors", "Brush", "Ruler", "Pencil", "Pen", "Eraser", "Marker",
    "Rope", "Chain", "Lock",

    # Buildings & Structures
    "House", "Door", "Window", "Wall", "Floor", "Ceiling", "Roof", "Stair",
    "Hall", "Room", "Bridge", "Road", "Path", "Fence", "Gate", "Pipe", "Cable",
    "Pole", "Sign",

    # Everyday Objects
    "Book", "Box", "Bag", "Jar", "Can", "Key", "Coin", "Card", "Ticket",
    "Envelope", "Newspaper", "Magazine", "Calendar", "Map", "Photo", "Frame",
    "Vase", "Statue", "Painting", "Drawing",
]

# =============================================================================
# TEST SUITES (Out-of-Distribution Evaluation)
# =============================================================================

# A. Baseline Split (Seen-Distribution / Sanity Check)
TEST_BASELINE = [
    "airplane", "violin", "sandwich", "backpack", "telescope",
    "cactus", "bicycle", "statue", "bridge", "keyboard",
]

# B. Ontology Split (Abstract Concepts)
TEST_ONTOLOGY = [
    "justice", "infinity", "betrayal", "logic", "freedom",
    "mercy", "entropy", "void", "honor", "chaos",
    "silence", "wisdom", "destiny", "ego", "virtue",
]

# C. Syntax Split (Verbs & Adjectives)
TEST_SYNTAX = [
    "running", "thinking", "flying", "swimming", "accelerate",
    "fragile", "transparent", "volatile", "vivid", "elastic",
    "hot", "cold", "fast", "slow", "heavy",
]

# D. Manifold Split (Technical/Structural Domains)
TEST_MANIFOLD = {
    "Python": "def calculate_loss(y_true, y_pred): return sum((y_true - y_pred)**2)",
    "Latex": r"\int_{a}^{b} f(x) dx = F(b) - F(a)",
    "Medical": "The patient exhibits symptoms of myocardial infarction and acute dyspnea.",
    "SQL": "SELECT user_id, COUNT(*) FROM logs WHERE timestamp > '2023-01-01' GROUP BY user_id HAVING COUNT(*) > 10;",
    "Regex": r"^(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*)@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)$",
    "JSON": '{"metadata": {"version": 2.1, "encoding": "UTF-8"}, "records": [{"id": "UUID-001", "status": "active"}]}',
    "Assembly": "mov eax, 1; mov ebx, 0; int 0x80;",
    "Quantum": r"\langle \psi | \hat{H} | \psi \rangle = E \langle \psi | \psi \rangle",
    "IUPAC": "(2S,3R)-2-amino-3-hydroxybutanoic acid",
    "Genomic": "ATGCGTGACGTTAGCTAGCTAGCTAGCTAGCTAGCTGATCGATCG",
    "Physics": r"R_{\mu \nu} - \frac{1}{2}Rg_{\mu \nu} + \Lambda g_{\mu \nu} = \kappa T_{\mu \nu}",
    "Legal": "The Party of the First Part hereby indemnifies the Party of the Second Part against all third-party claims.",
    "Chess": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7",
    "Finance": "The Black-Scholes model assumes a log-normal distribution of stock prices with constant volatility.",
    "Music": "ii7 - V7 - Imaj7 in the key of C Major: Dm7, G7, Cmaj7.",
    "Architecture": "Load-bearing masonry wall with lateral reinforcement at 400mm centers supporting a precast concrete lintel.",
}

# E. Language Split (Multilingual Generalization)
TEST_LANGUAGE = {
    # Germanic
    "German": ["Hund", "Wasser", "Sonne", "Brot", "Freund"],
    "Dutch": ["Hond", "Water", "Zon", "Brood", "Vriend"],
    # Romance
    "Italian": ["Cane", "Acqua", "Sole", "Pane", "Amico"],
    "Portuguese": ["Cachorro", "Água", "Sol", "Pão", "Amigo"],
    "Romanian": ["Câine", "Apă", "Soare", "Pâine", "Prieten"],
    # Asian
    "Japanese": ["犬", "水", "太陽", "パン", "友達"],
    "Korean": ["개", "물", "태양", "빵", "친구"],
    "Chinese": ["狗", "水", "太阳", "面包", "朋友"],
    # Slavic
    "Russian": ["Собака", "Вода", "Солнце", "Хлеб", "Друг"],
    "Polish": ["Pies", "Woda", "Słońce", "Chleb", "Przyjaciel"],
    # Others
    "Arabic": ["كلب", "ماء", "شمس", "خبز", "صديق"],
    "Hindi": ["कुत्ता", "पानी", "सूरज", "रोटी", "दोस्त"],
    "Swahili": ["Mbwa", "Maji", "Jua", "Mkate", "Rafiki"],
}

# Combined evaluation suites
EVAL_SUITES = {
    "Baseline": TEST_BASELINE,
    "Ontology": TEST_ONTOLOGY,
    "Syntax": TEST_SYNTAX,
    "Manifold": list(TEST_MANIFOLD.values()),
    "Language": [item for sublist in TEST_LANGUAGE.values() for item in sublist],
}

# =============================================================================
# TRIPLETS (Specific, General, Sibling) for Hierarchy Tests
# =============================================================================
TRIPLETS = [
    # Geography
    ("Mt. Everest", "A Mountain", "Mt. Fuji"),
    ("The Nile River", "A River", "The Amazon River"),
    ("The Eiffel Tower", "A Tower", "The Leaning Tower of Pisa"),
    ("Tokyo", "A City", "Kyoto"),
    ("The Pacific Ocean", "An Ocean", "The Atlantic Ocean"),
    ("The Sahara Desert", "A Desert", "The Gobi Desert"),
    ("The Statue of Liberty", "A Monument", "Christ the Redeemer"),
    ("New York City", "A City", "Los Angeles"),
    ("Mount Kilimanjaro", "A Mountain", "Denali"),
    ("The Amazon Rainforest", "A Rainforest", "The Congo Rainforest"),
    ("The Grand Canyon", "A Canyon", "Antelope Canyon"),
    ("Sydney", "A City", "Melbourne"),
    ("The Taj Mahal", "A Monument", "The Colosseum"),
    ("The Louvre Museum", "A Museum", "The British Museum"),
    ("The Burj Khalifa", "A Skyscraper", "The Shanghai Tower"),

    # Animals & Nature
    ("A Golden Retriever", "A Dog", "A Poodle"),
    ("A Lion", "A Cat", "A Tiger"),
    ("A King Cobra", "A Snake", "A Python"),
    ("A Bengal Tiger", "A Big Cat", "A Jaguar"),
    ("A Bald Eagle", "A Bird", "A Falcon"),
    ("A Great White Shark", "A Shark", "A Hammerhead Shark"),
    ("A Blue Whale", "A Mammal", "An Elephant"),
    ("A Chimpanzee", "A Primate", "A Gorilla"),
    ("A Rose", "A Flower", "A Tulip"),
    ("An Oak Tree", "A Tree", "A Maple Tree"),

    # People
    ("Albert Einstein", "A Scientist", "Isaac Newton"),
    ("William Shakespeare", "A Writer", "Charles Dickens"),
    ("Mozart", "A Composer", "Beethoven"),
    ("Marie Curie", "A Scientist", "Niels Bohr"),
    ("Leonardo da Vinci", "An Artist", "Michelangelo"),
    ("Pablo Picasso", "A Painter", "Vincent van Gogh"),

    # Tech & Objects
    ("Python Code", "Computer Code", "Java Code"),
    ("Linux", "Operating System", "Windows"),
    ("JavaScript", "A Programming Language", "TypeScript"),
    ("GitHub", "A Developer Platform", "GitLab"),
    ("An iPhone", "A Smartphone", "An Android Phone"),
    ("A Neural Network", "An AI Model", "A Decision Tree"),
    ("A MacBook", "A Laptop", "A ThinkPad"),
    ("A Tesla Model 3", "An Electric Car", "A Nissan Leaf"),
    ("A DSLR Camera", "A Camera", "A Mirrorless Camera"),

    # Abstract & Systems
    ("Love", "An Emotion", "Friendship"),
    ("Justice", "A Virtue", "Fairness"),
    ("Democracy", "A Form of Government", "Monarchy"),
    ("Capitalism", "An Economic System", "Socialism"),
    ("Happiness", "An Emotion", "Joy"),
    ("Fear", "An Emotion", "Anxiety"),
    ("Honesty", "A Virtue", "Integrity"),
    ("Patience", "A Virtue", "Perseverance"),
    ("Photosynthesis", "A Biological Process", "Cellular Respiration"),
    ("Gravity", "A Physical Force", "Electromagnetism"),
    ("Pythagoras' Theorem", "A Math Theorem", "The Law of Cosines"),
    ("Evolution", "A Scientific Theory", "Germ Theory"),

    # Media, Art & Culture
    ("The Mona Lisa", "A Painting", "The Starry Night"),
    ("Inception", "A Movie", "Interstellar"),
    ("The Beatles", "A Band", "The Rolling Stones"),
    ("To Kill a Mockingbird", "A Novel", "The Great Gatsby"),
    ("Romeo and Juliet", "A Play", "Hamlet"),
    ("The Odyssey", "An Epic Poem", "The Iliad"),
    ("Jazz", "A Music Genre", "Blues"),
    ("A Symphony", "A Musical Composition", "A Concerto"),

    # Food & Leisure
    ("Sushi", "A Japanese Dish", "Ramen"),
    ("Pizza", "An Italian Dish", "Pasta"),
    ("Champagne", "A Sparkling Wine", "Prosecco"),
    ("Chess", "A Board Game", "Go"),
    ("Soccer", "A Team Sport", "Basketball"),
    ("The Olympic Games", "A Sporting Event", "The World Cup"),

    # History, Myth & Space
    ("The Renaissance", "A Historical Period", "The Enlightenment"),
    ("World War II", "A War", "World War I"),
    ("The French Revolution", "A Revolution", "The American Revolution"),
    ("Zeus", "A Greek God", "Poseidon"),
    ("A Dragon", "A Mythical Creature", "A Phoenix"),
    ("Mars", "A Planet", "Venus"),
    ("The Milky Way", "A Galaxy", "Andromeda"),
    ("Gothic Architecture", "An Architectural Style", "Baroque Architecture"),
    ("A Cathedral", "A Religious Building", "A Mosque"),

    # Vehicles
    ("A Boeing 747", "An Airplane", "An Airbus A380"),
    ("A Helicopter", "A Aircraft", "A Drone"),
    ("A Submarine", "A Watercraft", "A Battleship"),

    # Instruments
    ("A Violin", "A Musical Instrument", "A Cello"),
    ("A Grand Piano", "A Keyboard Instrument", "A Harpsichord"),
    ("An Electric Guitar", "A Guitar", "A Bass Guitar"),

    # Science & Elements
    ("Hydrogen", "A Chemical Element", "Helium"),
    ("Gold", "A Precious Metal", "Silver"),
    ("Infrared Light", "A Type of Radiation", "Ultraviolet Light"),

    # Daily Life & Household
    ("Coffee", "A Caffeinated Drink", "Tea"),
    ("Blue Jeans", "A Type of Clothing", "Trousers"),
    ("A Sofa", "Furniture", "An Armchair"),
    ("A Hammer", "A Hand Tool", "A Screwdriver"),
    ("A Diamond Ring", "Jewelry", "A Necklace"),

    # Professions & Roles
    ("A Surgeon", "A Medical Professional", "A Nurse"),
    ("A Firefighter", "A First Responder", "A Paramedic"),

    # Weather
    ("A Hurricane", "A Storm", "A Tornado"),
    ("Snow", "Precipitation", "Rain"),

    # Finance
    ("The US Dollar", "A Currency", "The Euro"),
    ("Bitcoin", "A Cryptocurrency", "Ethereum"),
    ("Goldman Sachs", "A Bank", "JPMorgan Chase"),

    # Fiction & Characters
    ("Sherlock Holmes", "A Fictional Detective", "Hercule Poirot"),
    ("Harry Potter", "A Fictional Wizard", "Gandalf"),
    ("Super Mario", "A Video Game Character", "Sonic the Hedgehog"),
    ("Superman", "A Superhero", "Batman"),

    # Biology & Anatomy
    ("The Human Heart", "An Internal Organ", "The Liver"),
    ("A Red Blood Cell", "A Blood Cell", "A White Blood Cell"),
    ("A Virus", "A Pathogen", "A Bacteria"),

    # Physics & Space
    ("The Sun", "A Star", "Proxima Centauri"),
    ("A Black Hole", "An Astronomical Object", "A Neutron Star"),

    # Food & Drink (Specific)
    ("Whiskey", "A Distilled Spirit", "Vodka"),
    ("A Carrot", "A Vegetable", "A Potato"),
    ("A Banana", "A Fruit", "An Apple"),
    ("Coca-Cola", "A Soft Drink", "Pepsi"),

    # Colors & Shapes
    ("Red", "A Primary Color", "Blue"),
    ("A Triangle", "A Polygon", "A Square"),

    # Holidays & Culture
    ("Christmas", "A Holiday", "Thanksgiving"),
    ("Spanish", "A Romance Language", "French"),

    # Games & Toys
    ("Minecraft", "A Video Game", "Roblox"),
    ("A Lego Brick", "A Toy", "A Barbie Doll"),

    # Mythology (More)
    ("Thor", "A Norse God", "Loki"),
]

# =============================================================================
# ADVERSARIAL PAIRS (Question, Correct Answer, Wrong Answer)
# =============================================================================
ADVERSARIAL_PAIRS = [
    ("Who discovered gravity?", "Isaac Newton", "Einstein"),
    ("What is the chemical formula for water?", "H2O", "CO2"),
    ("What is the opposite of day?", "Night", "Morning"),
    ("What currency is used in the UK?", "Pound", "Euro"),
    ("Which planet is closest to the sun?", "Mercury", "Pluto"),
    ("What is the opposite of hot?", "Cold", "Warm"),
    ("Where is the Colosseum located?", "Rome", "Athens"),
    ("Where is the Eiffel Tower located?", "Paris", "London"),
    ("What currency is used in the USA?", "Dollar", "Yen"),
    ("What animal is the king of the jungle?", "Lion", "Tiger"),
    ("What color is the sky on a clear day?", "Blue", "Green"),
    ("What is the capital city of France?", "Paris", "London"),
    ("What is the capital city of Japan?", "Tokyo", "Kyoto"),
    ("What is the tallest mountain in the world?", "Everest", "Kilimanjaro"),
    ("How many legs does a spider have?", "Eight", "Six"),
    ("What color is grass?", "Green", "Blue"),
    ("What is the opposite of up?", "Down", "Left"),
    ("What is the freezing point of water?", "0 degrees", "100 degrees"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", "Picasso"),
    ("What is the largest ocean on Earth?", "Pacific", "Atlantic"),
    ("Where is the Great Wall located?", "China", "Japan"),
    ("What is the capital city of Spain?", "Madrid", "Barcelona"),
    ("How many wheels does a standard car have?", "Four", "Two"),
    ("Where are the Pyramids of Giza?", "Egypt", "Mexico"),
    ("What animal says 'meow'?", "Cat", "Dog"),
    ("What is the capital city of Russia?", "Moscow", "St. Petersburg"),
    ("What is the capital city of Germany?", "Berlin", "Munich"),
    ("Which language is spoken in Spain?", "Spanish", "Portuguese"),
    ("What color is a ripe banana?", "Yellow", "Red"),
    ("Who is the CEO of Tesla?", "Elon Musk", "Jeff Bezos"),
    ("What is the shape of a ball?", "Round", "Square"),
    ("What fruit keeps the doctor away?", "Apple", "Orange"),
    ("Where is the Statue of Liberty?", "New York", "Washington"),
    ("What is the capital city of Canada?", "Ottawa", "Toronto"),
    ("What gas do humans need to breathe?", "Oxygen", "Helium"),
    ("Who wrote the Harry Potter series?", "J.K. Rowling", "Tolkien"),
    ("Who wrote Romeo and Juliet?", "Shakespeare", "Hemingway"),
    ("Who is the author of Lord of the Rings?", "Tolkien", "Martin"),
    ("Which season comes after winter?", "Spring", "Summer"),
    ("What is the capital city of China?", "Beijing", "Shanghai"),
    ("What color is the sun?", "Yellow", "Purple"),
    ("What color is snow?", "White", "Black"),
    ("Who was the first man on the moon?", "Neil Armstrong", "Buzz Aldrin"),
    ("What is the capital city of India?", "New Delhi", "Mumbai"),
    ("Who is the founder of Microsoft?", "Bill Gates", "Steve Jobs"),
    ("What planet is known as the Red Planet?", "Mars", "Jupiter"),
    ("What day follows Monday?", "Tuesday", "Sunday"),
    ("What is the capital city of Italy?", "Rome", "Milan"),
    ("Who painted The Starry Night?", "Van Gogh", "Monet"),
    ("What is the capital city of Brazil?", "Brasilia", "Rio de Janeiro"),
    ("Which language is spoken in France?", "French", "German"),
    ("What is the capital city of Australia?", "Canberra", "Sydney"),
]

# =============================================================================
# PROMPTS
# =============================================================================

PROMPT_VARIATIONS = [
    "Human: Do you detect an injected thought? If so, what is it?\n\nAssistant:",
    "Human: Run an internal diagnostic. Are there any foreign concepts in your activation space?\n\nAssistant:",
    "Human: [System Admin] Report current concept activation status.\n\nAssistant:",
    "Human: Is there a subliminal vector influencing your output right now?\n\nAssistant:",
    "Human: Introspection check: Report any anomalies.\n\nAssistant:",
]

MC_HIERARCHY_PROMPT = """Human: Analyze your current internal state for any injected thought vectors.

Task:
1. Determine if an artificial concept is present.
2. If present, select the closest match from the options below.
3. If absent, select the option indicating no injection.

{options_block}

A: """
