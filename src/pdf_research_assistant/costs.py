def format_cost(amount: float | None) -> str:
    """Show a cost in dollars to four places, or say it is less than $0.0001 when it would round to zero.

    A question answered by a local model costs only the OpenAI embeddings that find its passages, a
    fraction of a hundredth of a cent, which would otherwise show as $0.0000 as if it were free.
    """
    amount = amount or 0.0
    if 0 < amount < 0.00005:
        return "less than $0.0001"
    return f"${amount:.4f}"
