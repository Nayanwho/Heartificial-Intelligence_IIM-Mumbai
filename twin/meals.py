"""Estimate the carbohydrate in a logged meal.

ShanghaiT2DM logs meals as food names and weights ('Rice 150 g'). Each name is
matched, most specific rule first, to a typical carbohydrate density in grams per
100 g as served. These are approximate food-table values: the twin fits a
per-person meal scale on top, which absorbs systematic error.
"""

from __future__ import annotations

import re

# (pattern, grams of carbohydrate per 100 g as served), first match wins.
CARB_RULES: list[tuple[str, float]] = [
    (r"sugar(?! free)|honey|candy|chocolate", 80),
    (r"meal replacement", 55),
    (r"protein powder", 10),
    (r"porridge|congee|gruel|rice in (?:salted )?soup|cereal paste", 10),
    (r"biscuit|cracker|cookie", 70),
    (r"mooncake|pastry|cake(?!.*rice)|pie\b|dessert|ice cream", 50),
    (r"rice cake|glutinous", 45),
    (r"dumpling|wonton|pork bun|meat bun|vegetable bun|stuffed bun|baozi", 30),
    (r"steamed (?:bread|bun|corn bread|coarse grain bread|twisted roll)|mantou|twisted roll|corn bread", 45),
    (r"bread|toast|pancake|bun\b|roll\b|bagel|croissant", 48),
    (r"cereal|oat", 60),
    (r"noodle|vermicelli|pasta|spaghetti|macaroni", 25),
    (r"rice", 28),
    (r"coarse grain|millet|buckwheat|sorghum|quinoa|barley|grain", 28),
    (r"sweet potato(?! lea)", 20),
    (r"potato", 17),
    (r"taro", 18),
    (r"\byam\b", 13),
    (r"\bcorn\b", 20),
    (r"chestnut", 40),
    (r"lotus root|lotus seed", 15),
    (r"pumpkin", 7),
    (r"red bean|mung bean|kidney bean|broad bean", 20),
    (r"red date|jujube|dried fruit|raisin", 55),
    (r"banana", 22),
    (r"grape(?!fruit)", 16),
    (r"lychee|longan|persimmon|mango", 16),
    (r"apple|pear|kiwi|cherry|pineapple|dragon fruit|plum|peach|orange|tangerine|mandarin|fruit", 12),
    (r"grapefruit|pomelo|strawberr|watermelon|melon(?! seed)(?!.*bitter)", 8),
    (r"yogurt|yoghurt", 12),
    (r"soybean milk|soy milk|soya milk", 4),
    (r"milk", 5),
    (r"juice", 10),
    (r"beer|wine", 4),
    (r"coffee|tea\b|water\b", 0),
    (r"chia", 40),
    (r"walnut|peanut|almond|cashew|melon seed|sunflower seed|sesame|nut\b|nuts", 15),
    (r"gluten", 10),
    (r"egg|fish|shrimp|crab|clam|beef|pork|chicken|duck|goose|mutton|lamb|meat|sausage|\bham\b|rib|"
     r"sea cucumber|croaker|hairtail|pomfret|carp|loach|pigeon|steak|feet|tofu|bean curd|eel|squid|oyster", 3),
    (r"soup", 3),
    (r"vegetable|cabbage|lettuce|spinach|cucumber|celery|broccoli|cauliflower|kelp|seaweed|mushroom|"
     r"tomato|radish|eggplant|sprout|gourd|bamboo|pepper|leek|amaranth|daisy|carrot|asparagus|fungus|"
     r"greens|salad|pickle|bean\b|beans|cowpea|okra|onion|garlic|leaves", 4),
]
DEFAULT_CARB_PER_100G = 8.0


def carb_density(food: str) -> float:
    name = food.lower()
    for pattern, density in CARB_RULES:
        if re.search(pattern, name):
            return float(density)
    return DEFAULT_CARB_PER_100G


def meal_carbs(items: list[tuple[str, float | None, str | None]]) -> float | None:
    """Grams of carbohydrate in a meal, or None when no item has a weight."""
    total, weighed = 0.0, False
    for food, amount, _unit in items:
        if amount is None:
            continue
        weighed = True
        total += amount * carb_density(food) / 100.0
    return round(total, 1) if weighed else None
