import argparse
from collections import deque
import sys
from typing import List, Tuple, Set


def read_conllu(path: str):
    """Yield sentences.  Each sentence is a list of tuples
    (id, form, head, deprel) where id is 1‑based."""
    sent = []
    with open(path, encoding='utf‑8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    yield sent
                    sent = []
                continue
            if line.startswith('#'):
                continue  # Comment
            fields = line.split('\t')
            if '-' in fields[0] or '.' in fields[0]:
                # skip multi‑word token / empty node
                continue
            id_ = int(fields[0])
            form = fields[1]
            head = int(fields[6])
            deprel = fields[7]
            sent.append((id_, form, head, deprel))
    if sent:
        yield sent


def oracle_arc_standard(head: List[int], label: List[str]) -> List[str]:
    """Return action strings for one sentence."""
    n = len(head) - 1
    stack = [0]  # 0 => ROOT
    buffer = deque(range(1, n + 1))
    arcs: List[Tuple[int, int, str]] = []
    actions: List[str] = []

    # gold children map
    gold_children: List[Set[int]] = [set() for _ in range(n + 1)]
    for d in range(1, n + 1):
        h = head[d]
        if h >= 0:
            gold_children[h].add(d)

    def built_children(node: int):
        return {d for (h, d, _) in arcs if h == node}

    while buffer or len(stack) > 1:
        if len(stack) >= 2:
            s0, s1 = stack[-1], stack[-2]

            # LEFT‑ARC: s0 is head of s1
            if head[s1] == s0 and gold_children[s1] <= built_children(s1):
                actions.append(f"LEFT‑ARC({label[s1]})")
                arcs.append((s0, s1, label[s1]))
                stack.pop(-2)
                continue

            # RIGHT‑ARC: s1 is head of s0
            if head[s0] == s1 and gold_children[s0] <= built_children(s0):
                actions.append(f"RIGHT‑ARC({label[s0]})")
                arcs.append((s1, s0, label[s0]))
                stack.pop()
                continue

        # SHIFT
        w = buffer.popleft()
        stack.append(w)
        actions.append("SHIFT")

    return actions


def process_sentence(sent, verbose=False):
    n = len(sent)
    # index 0 is ROOT
    head = [-1] + [None] * n
    label = [""] + [None] * n
    forms = ["ROOT"] + [None] * n
    for (idx, form, h, rel) in sent:
        head[idx] = h
        label[idx] = rel
        forms[idx] = form

    actions = oracle_arc_standard(head, label)

    if verbose:
        print("Forms:", forms)
        print("Heads:", head)
        print("Labels:", label)
        for i, a in enumerate(actions, 1):
            print(f"{i:02d}\t{a}")
    return actions


def main():
    ap = argparse.ArgumentParser(
        description="Generate gold action sequences for Arc‑Standard transition system.")
    ap.add_argument('conllu', help='Input file (CoNLL‑U)')
    ap.add_argument('-o', '--output', help='Write actions to file')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    all_actions = []
    for sent_idx, sent in enumerate(read_conllu(args.conllu), 1):
        actions = process_sentence(sent, args.verbose)
        all_actions.append(actions)
        if not args.output and not args.verbose:
            print(f"Sentence {sent_idx}: {len(actions)} actions")

    if args.output:
        with open(args.output, 'w', encoding='utf‑8') as fo:
            for sent_idx, actions in enumerate(all_actions, 1):
                fo.write(f"# sent_id = {sent_idx}\n")
                for act in actions:
                    fo.write(act + "\n")
                fo.write("\n")
        print(f"Action sequences written to {args.output}")


if __name__ == '__main__':
    main()
