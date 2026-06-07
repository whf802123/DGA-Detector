from pathlib import Path
import csv
import tldextract


PRIVATE_TLD_PATH = Path("private_tld.txt")
TRAINING_PATH = Path("training_w_tld.txt")
NGRAM_RANK_PATH = Path("n_gram_rank_freq.txt")
OUTPUT_PATH = Path("gram_ranks_training.txt")
UNKNOWN_RANK = 0

def average(values):
    return sum(values) / len(values) if values else 0

def generate_ngrams(text, n):
    for i in range(len(text) - n + 1):
        yield text[i:i + n]

def load_private_tlds(file_path):
    with file_path.open("r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}

def load_ngram_ranks(file_path):
    rank_dict = {}

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.rstrip("\n").split(",")

            if len(parts) != 4:
                continue

            category, gram, frequency, rank = parts

            try:
                rank_dict[gram] = int(rank)
            except ValueError:
                continue

    return rank_dict

def extract_core_domain(domain, private_tlds):
    extracted = tldextract.extract(domain)

    if extracted.domain.startswith("xn--"):
        return None

    core_domain = extracted.domain
    suffix = extracted.suffix

    if suffix in private_tlds:
        suffix_parts = suffix.split(".")

        if len(suffix_parts) >= 2:
            core_domain = suffix_parts[-2]
            suffix = suffix_parts[-1]

    if not core_domain:
        return None

    return f"${core_domain}$"

def calculate_ngram_average_rank(text, n, rank_dict, unknown_rank=UNKNOWN_RANK):
    ranks = [
        rank_dict.get(ngram, unknown_rank)
        for ngram in generate_ngrams(text, n)
    ]

    return average(ranks)

def process_training_file(
    training_path,
    output_path,
    private_tlds,
    ngram_ranks,
):
    with training_path.open("r", encoding="utf-8") as input_file, \
         output_path.open("w", encoding="utf-8", newline="") as output_file:

        writer = csv.writer(output_file)
        writer.writerow(["domain", "class", "s1", "s2", "s3", "core"])

        for line_number, line in enumerate(input_file, start=1):
            parts = line.rstrip("\n").split("\t")

            if len(parts) != 3:
                continue

            domain, domain_class, original_tld = parts

            core = extract_core_domain(domain, private_tlds)

            if core is None:
                continue

            unigram_text = core[1:-1]

            unigram_score = calculate_ngram_average_rank(
                unigram_text,
                1,
                ngram_ranks,
            )

            bigram_score = calculate_ngram_average_rank(
                core,
                2,
                ngram_ranks,
            )

            trigram_score = calculate_ngram_average_rank(
                core,
                3,
                ngram_ranks,
            )

            writer.writerow([
                domain,
                domain_class,
                f"{unigram_score:.2f}",
                f"{bigram_score:.2f}",
                f"{trigram_score:.2f}",
                core,
            ])

def main():
    private_tlds = load_private_tlds(PRIVATE_TLD_PATH)
    ngram_ranks = load_ngram_ranks(NGRAM_RANK_PATH)

    process_training_file(
        training_path=TRAINING_PATH,
        output_path=OUTPUT_PATH,
        private_tlds=private_tlds,
        ngram_ranks=ngram_ranks,
    )

if __name__ == "__main__":
    main()
  
