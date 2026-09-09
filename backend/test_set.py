# Each entry: a realistic question, and the page number(s) known to contain the actual answer.
# We built this by reading the document ourselves - this is the "ground truth."

TEST_QUERIES = [
    {
        "question": "What is the difference between large-scale and small-scale mining?",
        "relevant_pages": [1, 16],
    },
    {
        "question": "What environmental damage does mining cause?",
        "relevant_pages": [4],
    },
    {
        "question": "How do artisanal miners live day to day?",
        "relevant_pages": [3, 16],
    },
    {
        "question": "How are gemstones valued and traded?",
        "relevant_pages": [8],
    },
    {
        "question": "What role do anthropologists play in studying mining communities?",
        "relevant_pages": [1, 7],
    },
]