"""Light augmentation: swap, deletion, synonym replace."""
import random
from typing import List

STOPWORDS = {"the","a","an","is","are","was","were","and","or","of","to","in",
             "na","ya","ni","kwa","wa"}  # add Swahili/Hausa/Yoruba stopwords as needed

def random_swap(words, p=0.05):
    if len(words) < 2: return words
    out = words[:]
    for _ in range(max(1, int(len(out) * p))):
        i, j = random.sample(range(len(out)), 2)
        out[i], out[j] = out[j], out[i]
    return out

def random_deletion(words, p=0.05):
    if len(words) == 1: return words
    out = [w for w in words if random.random() > p]
    return out or [random.choice(words)]

def augment_text(text, synonym_replace_p=0.10,
                 random_swap_p=0.05, random_deletion_p=0.05):
    words = text.split()
    words = random_swap(words, random_swap_p)
    words = random_deletion(words, random_deletion_p)
    return " ".join(words)
