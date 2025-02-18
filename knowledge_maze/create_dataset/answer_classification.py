"""Classify answers to the relevant categories"""
import re
import spacy
import wikipedia
from dateutil.parser import parse

# Load spaCy's Named Entity Recognizer
nlp = spacy.load("en_core_web_sm")


def get_wikipedia_entity(answer):
    """Check if the answer is a valid Wikipedia entity."""
    try:
        page = wikipedia.page(answer)
        return page.title  # Return canonical Wikipedia entity
    except:
        try:
            page = wikipedia.page(answer, auto_suggest=False)
            return page.title
        except:
            return None  # Not a Wikipedia entity


def is_wikipedia_entity(answer):
    """Determine if an answer is a Wikipedia entity using NER and Wikipedia API."""
    doc = nlp(answer)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "NORP", "EVENT"}:
            entity = get_wikipedia_entity(answer)
            if entity:
                return entity  # Valid Wikipedia entity
    return None  # Not an entity


def is_number(answer):
    """Check if the answer is a number (integer, decimal, fraction)."""
    return bool(re.match(r"^\d+(\.\d+)?$", answer))


def is_numeric_entity(answer):
    """Check for numbers using spaCy NER."""
    doc = nlp(answer)
    return any(ent.label_ in {"CARDINAL", "QUANTITY", "PERCENT", "MONEY"} for ent in doc.ents)


def is_date(answer):
    """Check if the answer is a date using regex and parsing."""
    try:
        parse(answer, fuzzy=False)
        return True

    except:
        return False


def is_date_entity(answer):
    """Check for dates using spaCy NER."""
    doc = nlp(answer)
    return any(ent.label_ == "DATE" for ent in doc.ents)


def classify_answer(answer):
    """Classify an answer into Wikipedia Entity, Number, Date, or Free-form text."""
    entity = is_wikipedia_entity(answer)
    if entity:
        return "Wikipedia Entity", entity
    if is_number(answer) or is_numeric_entity(answer):
        return "Number", answer
    if is_date(answer) or is_date_entity(answer):
        return "Date", answer
    return "Free-form", answer  # Default case
