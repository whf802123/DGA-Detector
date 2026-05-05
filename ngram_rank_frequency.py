from collections import Counter
from pathlib import Path
import csv
import tldextract


PRIVATE_TLD_PATH = Path("private_tld.txt")
INPUT_PATH = Path("top-100k.csv")
OUTPUT_PATH = Path("n_gram_rank_freq.txt")


def generate_ngrams(text, n):
    """Generate character-level n-grams from text."""
    for index in range(len(text) - n + 1):
        yield text[index:index + n]


def load_private_tlds(file_path):
    """Load private TLDs from a plain text file."""
    with file_path.open("r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def extract_core_domain(domain, private_tlds):
    """
    Extract the core domain using tldextract.

    Example:
        www.google.com -> $google$

    If the suffix is a private TLD, this keeps the original script's logic:
        abc.blogspot.com -> $blogspot$
    """
    extracted = tldextract.extract(domain)

    if extracted.domain.startswith("xn--"):
        return None

    core_domain = extracted.domain
    suffix = extracted.suffix

    if suffix in private_tlds:
        suffix_parts = suffix.split(".")

        if len(suffix_parts) >= 2:
            core_domain = suffix_parts[-2]

    if not core_domain:
        return None

    return f"${core_domain}$"


def update_ngram_counts(core_domain, unigram_counts, bigram_counts, trigram_counts):
    """Update unigram, bigram, and trigram frequency counters."""
    unigram_counts.update(generate_ngrams(core_domain[1:-1], 1))
    bigram_counts.update(generate_ngrams(core_domain, 2))
    trigram_counts.update(generate_ngrams(core_domain, 3))


def read_domains_from_csv(file_path):
    """
    Read domains from a CSV file.

    Expected input format:
        rank,domain

    Example:
        1,google.com
        2,youtube.com
    """
    with file_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) < 2:
                continue

            rank, domain = row[0].strip(), row[1].strip()

            # Skip header if present
            if rank.lower() == "rank" and domain.lower() == "domain":
                continue

            if not domain:
                continue

            yield domain


def write_ranked_ngrams(output_path, ngram_type, counts, writer):
    """Write n-gram frequencies sorted by descending frequency."""
    sorted_items = sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for rank, (ngram, frequency) in enumerate(sorted_items, start=1):
        writer.writerow([ngram_type, ngram, frequency, rank])


def build_ngram_rank_table(input_path, private_tld_path, output_path):
    """Build an n-gram frequency and rank table from popular domains."""
    private_tlds = load_private_tlds(private_tld_path)

    unigram_counts = Counter()
    bigram_counts = Counter()
    trigram_counts = Counter()

    for domain in read_domains_from_csv(input_path):
        core_domain = extract_core_domain(domain, private_tlds)

        if core_domain is None:
            continue

        update_ngram_counts(
            core_domain=core_domain,
            unigram_counts=unigram_counts,
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
        )

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        write_ranked_ngrams(output_path, 1, unigram_counts, writer)
        write_ranked_ngrams(output_path, 2, bigram_counts, writer)
        write_ranked_ngrams(output_path, 3, trigram_counts, writer)


def main():
    build_ngram_rank_table(
        input_path=INPUT_PATH,
        private_tld_path=PRIVATE_TLD_PATH,
        output_path=OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
