import re
from typing import List


def clean_spaces(text: str) -> str:
    """
    Cleans up unnecessary spaces in a string.

    Arguments:
        - text: str:
          The input string to be processed.

    Returns:
        - cleaned_text: str:
          The cleaned string with:
            * Multiple spaces replaced by a single space.
            * Leading and trailing spaces removed.
            * Spaces before punctuation removed.
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = re.sub(r"\s([?.!,;:])", r"\1", text)
    return text


def normalize_punctuation(text: str) -> str:
    """
    Normalizes punctuation in a string.

    Arguments:
        - text: str:
          The input string to be processed.

    Returns:
        - normalized_text: str:
          The string with:
            * Typographic quotation marks replaced by standard quotes.
            * Typographic apostrophes replaced by standard apostrophes.
    """
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[’‘]", "'", text)
    return text


def remove_unwanted_chars(text: str) -> str:
    """
    Removes unwanted characters from a string.

    Arguments:
        - text: str:
          The input string to be processed.

    Returns:
        - cleaned_text: str:
          The string with all non-alphanumeric characters removed,
          except for punctuation (.,!?;:).
    """
    text = re.sub(r"[^\w\s.,!?;:]", "", text)
    return text


def preprocess_translation_data(texts: List[str]) -> List[str]:
    """
    Preprocesses a list of strings for translation tasks.

    This function applies a pipeline of preprocessing steps, including:
      * Cleaning up unnecessary spaces.
      * Normalizing punctuation.
      * Removing unwanted characters.
      * Lowercasing all text.

    Arguments:
        - texts: List[str]:
          A list of strings to be processed.

    Returns:
        - preprocessed_texts: List[str]:
          A list of preprocessed strings, ready for translation tasks.
    """
    preprocessed_texts = [
        clean_spaces(
            normalize_punctuation(remove_unwanted_chars(line.lower()))
        )
        for line in texts
    ]
    return preprocessed_texts
