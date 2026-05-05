import csv
import math
import pickle
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import tldextract

import gib_detect_train


PRIVATE_TLD_PATH = Path("private_tld.txt")
NGRAM_RANK_PATH = Path("n_gram_rank_freq.txt")
TRAINING_PATH = Path("training_w_tld.txt")
TRANSITION_MATRIX_PATH = Path("trans_matrix.csv")
GIB_MODEL_PATH = Path("gib_model.pki")
OUTPUT_PATH = Path("features.txt")

HMM_LOG_PROB_FLOOR = -999.0
HMM_LOG_PROB_THRESHOLD = -120.0

BOUNDARY_START = "$"
BOUNDARY_END = "$"
HMM_START = "^"
HMM_END = "$"

VOWELS = set("aeiou")
DIGITS = set("0123456789")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")


FEATURE_HEADER = [
    "ip",
    "class",
    "tld",
    "entropy",
    "len",
    "norm_entropy",
    "vowel_ratio",
    "digit_ratio",
    "repeat_letter",
    "consec_digit",
    "consec_consonant",
    "gib_value",
    "hmm_log",
    "uni_rank",
    "bi_rank",
    "tri_rank",
    "uni_std",
    "bi_std",
    "tri_std",
    "private_tld",
]


def safe_mean(values):
    """Return the mean of a numeric sequence. Return 0 for an empty sequence."""
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if values.size else 0.0


def safe_std(values):
    """Return the standard deviation of a numeric sequence. Return 0 for an empty sequence."""
    values = np.asarray(values, dtype=float)
    return float(values.std()) if values.size else 0.0


def generate_ngrams(text, n):
    """Generate character-level n-grams."""
    for index in range(len(text) - n + 1):
        yield text[index:index + n]


def count_vowels(text):
    """Count vowels in text."""
    return sum(1 for char in text.lower() if char in VOWELS)


def count_digits(text):
    """Count digits in text."""
    return sum(1 for char in text if char in DIGITS)


def count_repeated_letters(text):
    """
    Count how many distinct alphabetic characters appear more than once.

    Example:
        google -> g and o are repeated -> 2
    """
    letter_counts = Counter(char for char in text.lower() if char.isalpha())
    return sum(1 for count in letter_counts.values() if count > 1)


def count_consecutive_digits(text):
    """Count characters that are part of digit runs with length greater than 1."""
    digit_flags = [char.isdigit() for char in text]
    grouped_runs = groupby(digit_flags)

    return sum(
        run_length
        for is_digit, group in grouped_runs
        for run_length in [len(list(group))]
        if is_digit and run_length > 1
    )


def count_consecutive_consonants(text):
    """Count characters that are part of consonant runs with length greater than 1."""
    consonant_flags = [char.lower() in CONSONANTS for char in text]
    grouped_runs = groupby(consonant_flags)

    return sum(
        run_length
        for is_consonant, group in grouped_runs
        for run_length in [len(list(group))]
        if is_consonant and run_length > 1
    )


def calculate_entropy(text):
    """Calculate Shannon entropy using natural logarithm."""
    if not text:
        return 0.0

    length = float(len(text))
    counts = Counter(text)

    return -sum(
        count / length * math.log(count / length)
        for count in counts.values()
    )


def load_private_tlds(file_path):
    """Load private TLDs from a plain text file."""
    with file_path.open("r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def load_ngram_ranks(file_path):
    """
    Load n-gram rank dictionary.

    Expected input format:
        n,gram,frequency,rank
    """
    rank_dict = {}

    with file_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) != 4:
                continue

            _, gram, _, rank = row

            try:
                rank_dict[gram] = int(rank)
            except ValueError:
                continue

    return rank_dict


def load_transition_matrix(file_path):
    """
    Load Markov transition probabilities.

    Expected input format:
        current_gram<TAB>next_gram<TAB>probability
    """
    transitions = defaultdict(dict)

    with file_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            if len(row) != 3:
                continue

            current_gram, next_gram, probability = row

            try:
                transitions[current_gram][next_gram] = float(probability)
            except ValueError:
                continue

    return transitions


def load_gibberish_model(file_path):
    """Load gibberish detection model."""
    with file_path.open("rb") as file:
        model_data = pickle.load(file)

    return model_data["mat"], model_data["thresh"]


def extract_domain_parts(domain, private_tlds):
    """
    Extract core domain and suffix.

    This preserves the original private TLD handling:

        abc.blogspot.com -> core = $blogspot$, tld = com

    Returns:
        core_domain, tld, has_private_tld
    """
    extracted = tldextract.extract(domain)

    if extracted.domain.startswith("xn--"):
        return None, None, None

    core_label = extracted.domain
    suffix = extracted.suffix
    has_private_tld = 0

    if suffix in private_tlds:
        suffix_parts = suffix.split(".")

        if len(suffix_parts) >= 2:
            has_private_tld = 1
            core_label = suffix_parts[-2]
            suffix = suffix_parts[-1]

    if not core_label:
        return None, None, None

    core_domain = f"{BOUNDARY_START}{core_label}{BOUNDARY_END}"

    return core_domain, suffix, has_private_tld


def calculate_ngram_ranks(core_domain, rank_dict):
    """Calculate unigram, bigram, and trigram rank arrays."""
    unigram_ranks = [
        rank_dict.get(ngram, 0)
        for ngram in generate_ngrams(core_domain[1:-1], 1)
    ]

    bigram_ranks = [
        rank_dict.get(ngram, 0)
        for ngram in generate_ngrams(core_domain, 2)
    ]

    trigram_ranks = [
        rank_dict.get(ngram, 0)
        for ngram in generate_ngrams(core_domain, 3)
    ]

    return unigram_ranks, bigram_ranks, trigram_ranks


def calculate_markov_log_probability(domain, transitions):
    """
    Calculate bigram Markov log probability.

    The original script multiplied probabilities directly, which can underflow.
    This version sums log probabilities instead.

    If a transition is missing or probability is zero, return HMM_LOG_PROB_FLOOR.
    """
    hmm_domain = f"{HMM_START}{domain.strip('.')}{HMM_END}"
    bigrams = list(generate_ngrams(hmm_domain, 2))

    if not bigrams:
        return HMM_LOG_PROB_FLOOR

    log_probability = 0.0

    first_probability = transitions.get("", {}).get(bigrams[0], 0.0)

    if first_probability <= 0:
        return HMM_LOG_PROB_FLOOR

    log_probability += math.log(first_probability)

    for current_bigram, next_bigram in zip(bigrams, bigrams[1:]):
        probability = transitions.get(current_bigram, {}).get(next_bigram, 0.0)

        if probability <= 0:
            return HMM_LOG_PROB_FLOOR

        log_probability += math.log(probability)

        if log_probability < HMM_LOG_PROB_THRESHOLD:
            return HMM_LOG_PROB_FLOOR

    return log_probability


def extract_features(
    domain,
    domain_class,
    private_tlds,
    ngram_ranks,
    transitions,
    gib_model_matrix,
    gib_threshold,
):
    """Extract all features for one domain."""
    core_domain, suffix, has_private_tld = extract_domain_parts(domain, private_tlds)

    if core_domain is None:
        return None

    domain_length = float(len(core_domain))

    entropy = calculate_entropy(core_domain)
    norm_entropy = entropy / domain_length if domain_length else 0.0

    unigram_ranks, bigram_ranks, trigram_ranks = calculate_ngram_ranks(
        core_domain,
        ngram_ranks,
    )

    vowel_ratio = count_vowels(core_domain) / domain_length
    digit_ratio = count_digits(core_domain) / domain_length
    repeat_letter_ratio = count_repeated_letters(core_domain) / domain_length
    consecutive_digit_ratio = count_consecutive_digits(core_domain) / domain_length
    consecutive_consonant_ratio = count_consecutive_consonants(core_domain) / domain_length

    hmm_log_probability = calculate_markov_log_probability(domain, transitions)

    gib_value = int(
        gib_detect_train.avg_transition_prob(
            core_domain.strip("$"),
            gib_model_matrix,
        ) > gib_threshold
    )

    return [
        domain,
        domain_class,
        suffix,
        f"{entropy:.3f}",
        f"{domain_length:.1f}",
        f"{norm_entropy:.3f}",
        f"{vowel_ratio:.2f}",
        f"{digit_ratio:.2f}",
        f"{repeat_letter_ratio:.2f}",
        f"{consecutive_digit_ratio:.2f}",
        f"{consecutive_consonant_ratio:.2f}",
        f"{gib_value:.2f}",
        f"{hmm_log_probability:.2f}",
        f"{safe_mean(unigram_ranks):.2f}",
        f"{safe_mean(bigram_ranks):.2f}",
        f"{safe_mean(trigram_ranks):.2f}",
        f"{safe_std(unigram_ranks):.2f}",
        f"{safe_std(bigram_ranks):.2f}",
        f"{safe_std(trigram_ranks):.2f}",
        has_private_tld,
    ]


def read_training_rows(file_path):
    """
    Read training rows.

    Expected input format:
        domain<TAB>class<TAB>tld
    """
    with file_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            if len(row) != 3:
                continue

            domain, domain_class, _ = row

            if not domain:
                continue

            yield domain.strip(), domain_class.strip()


def write_features(output_path, rows):
    """Write extracted features to a TSV file."""
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(FEATURE_HEADER)
        writer.writerows(rows)


def main():
    private_tlds = load_private_tlds(PRIVATE_TLD_PATH)
    ngram_ranks = load_ngram_ranks(NGRAM_RANK_PATH)
    transitions = load_transition_matrix(TRANSITION_MATRIX_PATH)
    gib_model_matrix, gib_threshold = load_gibberish_model(GIB_MODEL_PATH)

    feature_rows = []

    for domain, domain_class in read_training_rows(TRAINING_PATH):
        row = extract_features(
            domain=domain,
            domain_class=domain_class,
            private_tlds=private_tlds,
            ngram_ranks=ngram_ranks,
            transitions=transitions,
            gib_model_matrix=gib_model_matrix,
            gib_threshold=gib_threshold,
        )

        if row is not None:
            feature_rows.append(row)

    write_features(OUTPUT_PATH, feature_rows)


if __name__ == "__main__":
    main()
