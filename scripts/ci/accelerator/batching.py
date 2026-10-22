"""Batch compatible cases using fixed weights and device-count priorities."""


def make_batches(inventories, *, target_seconds=120, max_cases=64, capacity=None):
    batches = []
    for selection, rows in inventories:
        groups = {}
        for row in rows:
            cards = row["cards"]
            oversized_skip = capacity is not None and cards > capacity and row.get("skip") is not None
            if oversized_skip:
                # Keep declared plans intact; a skipped worker only needs the
                # runner's full slice to preserve collection-time visibility.
                cards = capacity
            groups.setdefault((row["file"], cards, oversized_skip), []).append(row)
        for (_, cards, _), cases in groups.items():
            current = []
            estimated = 0.0
            for row in cases:
                # Fixed weights bound batch size; measurements do not affect dispatch.
                cost = 60.0 if cards > 1 else 2.0
                if current and (len(current) >= max_cases or estimated + cost > target_seconds):
                    batches.append(dict(selection=selection, cards=cards, cases=current, estimated_seconds=estimated))
                    current, estimated = [], 0.0
                current.append(row)
                estimated += cost
            if current:
                batches.append(dict(selection=selection, cards=cards, cases=current, estimated_seconds=estimated))
    # Larger reservations first; the device pool backfills with fitting jobs.
    # Stable ordering preserves collection order among equal reservations.
    batches.sort(key=lambda row: -row["cards"])
    for index, row in enumerate(batches):
        row["name"] = f"{row['selection']['name']}-batch-{index:04d}"
    return batches
