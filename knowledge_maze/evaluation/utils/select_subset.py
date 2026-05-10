import random
from collections import defaultdict


def get_stratified_subset(dataset, target_limit, strategy_key='Selected strategy'):
    """
    Returns a stratified subset of the dataset where each strategy is
    represented as equally possible.
    """
    # 1. Group items by strategy
    stratified_groups = defaultdict(list)
    for item in dataset:
        strat = item.get(strategy_key, 'Unknown')
        stratified_groups[strat].append(item)

    strategies = list(stratified_groups.keys())
    num_strats = len(strategies)

    if num_strats == 0:
        return []

    # 2. Calculate ideal samples per strategy
    per_strat_target = target_limit // num_strats
    remainder = target_limit % num_strats

    subset = []
    underfilled_slots = 0

    # 3. First Pass: Take samples from each group
    for i, strat in enumerate(strategies):
        group = stratified_groups[strat]
        # Distribute remainder across the first few strategies
        current_target = per_strat_target + (1 if i < remainder else 0)

        if len(group) >= current_target:
            subset.extend(random.sample(group, current_target))
        else:
            # If the group is too small, take everything and track the deficit
            subset.extend(group)
            underfilled_slots += (current_target - len(group))

    # 4. Second Pass: If some strategies were short, fill the remaining
    # slots from the leftover pool to hit the exact target_limit
    if underfilled_slots > 0:
        leftover_pool = [
            item for item in dataset
            if item not in subset
        ]
        if leftover_pool:
            extra_samples = random.sample(
                leftover_pool,
                min(len(leftover_pool), underfilled_slots)
            )
            subset.extend(extra_samples)

    # Final shuffle so the output isn't ordered by strategy
    random.shuffle(subset)

    # Optional: Log the distribution for verification
    print(f"✅ Stratified {len(subset)} samples across {num_strats} strategies.")
    return subset