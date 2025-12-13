# utils/lost_in_the_middle/metrics.py

This helper module implements evaluation utilities:

- `normalize_answer` replicates SQuAD-style normalization by lowercasing, stripping punctuation, removing articles, and collapsing whitespace.
- `best_subspan_em` checks whether any normalized gold answer appears as a substring of the normalized model prediction, returning 1.0 or 0.0 for binary exact-match scoring.
