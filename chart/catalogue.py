"""Robert's named Clone Hero patterns, encoded for machine checking.

Transcribed from docs/patterns-reference.md, which is AI-assisted and explicitly NOT
ground truth -- Robert flagged that entries may be wrong, mislabeled, nonexistent, or real
but with incorrect fingerings. The point of this module is to make those checkable: reduce
every named pattern to a transposition-invariant signature, then look for collisions within
the list and for occurrences in the corpus.

Fret order low to high is G R Y B O, encoded 0..4. Open notes are not part of any entry.
"""

from __future__ import annotations

FRETS = {"G": 0, "R": 1, "Y": 2, "B": 3, "O": 4}


def parse(sequence: str) -> tuple[int, ...]:
    """'G R Y B' or 'G-R-Y-B' -> (0, 1, 2, 3). Bar markers '|' and arrows are ignored."""
    cleaned = sequence.replace("-", " ").replace("|", " ").replace("->", " ")
    return tuple(FRETS[token] for token in cleaned.split() if token in FRETS)


def split_transition(sequence: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Separate the leading transition notes from the pattern body at the '|' marker.

    Robert: a transition note is whatever leads from one pattern into the next, so it
    belongs to the *join*, not to either pattern's identity. Which notes they are depends
    on the pair being joined, so a body must be matchable without them.

    Signing the whole written sequence -- transition included -- makes a pattern findable
    only where it happens to be preceded by the exact transition the reference author wrote
    down, which undercounts every entry that has one.

    Returns (transition, body). Entries with no marker are all body.
    """
    if "|" not in sequence:
        return (), parse(sequence)
    head, _, tail = sequence.partition("|")
    return parse(head), parse(tail)


def body_signature(sequence: str) -> tuple[int, ...]:
    """Transposition-invariant identity of the pattern body, transitions excluded."""
    return signature(split_transition(sequence)[1])


def signature(frets: tuple[int, ...]) -> tuple[int, ...]:
    """Transposition-invariant identity: consecutive differences."""
    return tuple(b - a for a, b in zip(frets, frets[1:]))


# name -> the example sequence as written in the reference
CATALOGUE: dict[str, str] = {
    # --- trip zigs -------------------------------------------------------------
    "trip zig G-R-Y": "Y R | G R Y R G R Y R G",
    "trip zig R-Y-B": "B Y | R Y B Y R Y B Y R",
    "trip zig Y-B-O": "O B | Y B O B Y B O B Y",
    "trip zig split G-Y-B": "B Y | G Y B Y G Y B Y G",
    "trip zig split G-R-B": "B R | G R B R G R B R G",
    "trip zig split R-B-O": "O B | R B O B R B O B R",
    "trip zig split G-Y-O": "O Y | G Y O Y G Y O Y G",
    "trip zig split G-R-O": "O R | G R O R G R O R G",
    "trip zig split R-Y-O": "O Y | R Y O Y R Y O Y R",
    "trip zig split G-B-O": "O B | G B O B G B O B G",
    # --- quad zigs -------------------------------------------------------------
    "quad zig G-R-Y-B": "B Y R | G R Y B Y R G R Y B",
    "quad zig R-Y-B-O": "O B Y | R Y B O B Y R Y B O",
    "quad zig split G-Y-B-O": "O B Y | G Y B O B Y G Y B O",
    "quad zig split G-R-B-O": "O B R | G R B O B R G R B O",
    "quad zig split G-R-Y-O": "O Y R | G R Y O Y R G R Y O",
    # --- quint zig -------------------------------------------------------------
    "quint zig G-R-Y-B-O": "O B Y R | G R Y B O B Y R G R Y B O",
    # --- chimneys --------------------------------------------------------------
    "trip chimney G-R-B, Y peak": "G R B R G Y",
    "trip chimney G-R-Y, B peak": "G R Y R G B",
    "trip chimney R-Y-B, O peak": "R Y B Y R O",
    "trip chimney R-Y-O, B peak": "R Y O Y R B",
    "trip chimney G-Y-B, O peak": "G Y B Y G O",
    "quad chimney G-R-Y-B, O peak": "G R Y B Y R G O",
    "quad chimney R-Y-B-O, O peak": "R Y B O B Y R O",
    "quad chimney G-R-Y-B, B peak": "G R Y B Y R G B",
    "quad chimney G-R-B-O, Y peak": "G R B O B R G Y",
    "quint chimney full neck": "G R Y B O B Y R G O",
    "reverse chimney trip a": "B Y R Y B R",
    "reverse chimney trip b": "O B Y B O Y",
    "reverse chimney quad": "O B Y R Y B O R",
    "reverse chimney quint": "O B Y R G R Y B O R",
    "reverse chimney quint to base": "O B Y R G R Y B O G",
    # --- cakes -----------------------------------------------------------------
    # Corrected from Robert's own sequences and confirmed in Hot N Cold, where the
    # 8-note appears as O G O G R Y R G and the 12-note as B G B G B G R Y B Y R G.
    # The previous entries described two trills back to back (G R G R O B O B), which
    # is not the shape: a cake trills against the anchor fret, repeats that, then
    # descends back to it. Written wrong, they never matched anything.
    "cake 6-note": "Y G Y G R G",
    "cake 8-note": "B G B G R Y R G",
    "cake 10-note": "B G B G B G R Y R G",
    "cake 12-note": "O G O G O G R Y B Y R G",
    "cake 14-note": "O G O G O G R Y B O B Y R G",
    # --- slides and H ----------------------------------------------------------
    "triangle slide 6-note": "G R Y B Y R",
    "triangle slide 8-note": "G R Y B O B Y R",
    "H pattern standard": "G O Y B Y G O",
    # --- sweeps ----------------------------------------------------------------
    "sweep 6-note": "G R Y B O B",
    "sweep 8-note": "G R Y B O B Y R",
    "sweep 12-note": "G R Y B O B Y R G R Y B",
    # --- trips -----------------------------------------------------------------
    "trip asc contiguous G-R-Y": "G R Y",
    "trip asc contiguous R-Y-B": "R Y B",
    "trip asc contiguous Y-B-O": "Y B O",
    "trip asc gapped G-Y-B": "G Y B",
    "trip asc gapped R-B-O": "R B O",
    "trip asc gapped G-R-B": "G R B",
    "trip asc gapped G-B-O": "G B O",
    "trip desc contiguous O-B-Y": "O B Y",
    "trip desc contiguous B-Y-R": "B Y R",
    "trip desc contiguous Y-R-G": "Y R G",
    "trip desc gapped O-B-R": "O B R",
    "trip desc gapped B-Y-G": "B Y G",
    "trip desc gapped O-Y-G": "O Y G",
    "trip desc gapped O-R-G": "O R G",
    # --- single-fret anchors ---------------------------------------------------
    "anchor G-R-G": "G R G",
    "anchor G-Y-G": "G Y G",
    "anchor G-B-G": "G B G",
    "anchor G-O-G": "G O G",
    "anchor R-G-R": "R G R",
    "anchor R-Y-R": "R Y R",
    "anchor R-B-R": "R B R",
    "anchor R-O-R": "R O R",
    "anchor Y-G-Y": "Y G Y",
    "anchor Y-R-Y": "Y R Y",
    "anchor Y-B-Y": "Y B Y",
    "anchor Y-O-Y": "Y O Y",
    "anchor B-G-B": "B G B",
    "anchor B-R-B": "B R B",
    "anchor B-Y-B": "B Y B",
    "anchor B-O-B": "B O B",
    "anchor O-B-O": "O B O",
    "anchor O-Y-O": "O Y O",
    "anchor O-R-O": "O R O",
    "anchor O-G-O": "O G O",
    # --- rake trills -----------------------------------------------------------
    "raked trill 2-finger": "O G O G O G",
    "raked trill 3-finger": "O G O G O G O G",
    # --- quads -----------------------------------------------------------------
    "quad asc contiguous G-R-Y-B": "G R Y B",
    "quad asc contiguous R-Y-B-O": "R Y B O",
    "quad adjacent overlapping": "G R Y B R Y B O",
    "quad asc gapped G-R-Y-O": "G R Y O",
    "quad asc gapped G-R-B-O": "G R B O",
    "quad asc gapped G-Y-B-O": "G Y B O",
    "quad desc contiguous O-B-Y-R": "O B Y R",
    "quad desc contiguous B-Y-R-G": "B Y R G",
    "quad desc gapped O-Y-R-G": "O Y R G",
    "quad desc gapped O-B-R-G": "O B R G",
    "quad desc gapped O-B-Y-G": "O B Y G",
    # --- quints ----------------------------------------------------------------
    "quint asc sweep": "G R Y B O",
    "quint desc sweep": "O B Y R G",
    # --- ladders ---------------------------------------------------------------
    "ladder descending split": "O Y B R Y G B R",
    "ladder wide desc": "O R B G",
    "ladder wide asc": "G B R O",
    "ladder triplet step asc": "G R Y R Y B Y B O",
    "ladder triplet step desc": "O B Y B Y R Y R G",
    # --- rolls -----------------------------------------------------------------
    "roll quad asc G-R-Y-B": "G R Y B G R Y B",
    "roll quad asc R-Y-B-O": "R Y B O R Y B O",
    "roll quad gapped G-Y-B-O": "G Y B O G Y B O",
    "roll quad gapped G-R-B-O": "G R B O G R B O",
    "roll quint asc": "G R Y B O G R Y B O G R Y B O",
    "roll quad desc B-Y-R-G": "B Y R G B Y R G",
    "roll quad desc O-B-Y-R": "O B Y R O B Y R",
    "roll quint desc": "O B Y R G O B Y R G O B Y R G",
    # --- trills ----------------------------------------------------------------
    "trill adjacent R-G": "R G R G",
    "trill adjacent Y-R": "Y R Y R",
    "trill adjacent B-Y": "B Y B Y",
    "trill adjacent O-B": "O B O B",
    "trill 1-split Y-G": "Y G Y G",
    "trill 1-split B-R": "B R B R",
    "trill 1-split O-Y": "O Y O Y",
    "trill 2-split B-G": "B G B G",
    "trill 2-split O-R": "O R O R",
    "trill full span O-G": "O G O G",
    # --- castles ---------------------------------------------------------------
    "castle green anchor (TTFAF)": "R G Y G B G O G B G Y G R G",
    "castle red anchor": "Y R B R O R B R Y R G R",
    "castle yellow anchor": "B Y O Y B Y R Y G Y R Y",
    "castle orange anchor (reverse)": "B O Y O R O G O R O Y O B O",
    "castle blue anchor (reverse)": "O B Y B R B G B R B Y B O B",
}


def signatures() -> dict[str, tuple[int, ...]]:
    """Every catalogued pattern reduced to its transposition-invariant signature."""
    return {name: signature(parse(text)) for name, text in CATALOGUE.items()}


def collisions() -> dict[tuple[int, ...], list[str]]:
    """Signatures claimed by more than one name.

    Some collisions are correct and expected -- transpositions of one shape share a
    signature by design. Others mean the list names one thing twice.
    """
    grouped: dict[tuple[int, ...], list[str]] = {}
    for name, sig in signatures().items():
        grouped.setdefault(sig, []).append(name)
    return {sig: names for sig, names in grouped.items() if len(names) > 1}


def chord_signature(positions: list[tuple[int, ...]]) -> tuple:
    """Transposition-invariant identity for a figure that may contain chords.

    `signature` only describes single notes, so any figure with a chord in it was
    invisible to pattern search -- which is how the H pattern came to be reported as
    never occurring when it is in 8.9% of charts. Robert: some patterns use tap chords.

    Each position splits into two parts:

    *shape* -- the frets relative to the lowest one in that position, so a single note
    is (0,) and RB and YO are both (0, 2). This is what the hand does.

    *root move* -- how far the lowest fret moved from the previous position. This is
    where the hand goes.

    So the H pattern RB -> RYB -> RB reads as shapes (0,2), (0,1,2), (0,2) with root
    moves 0, 0, and matches GO -> GYO -> GO at a different place on the neck, which is
    the same pattern and should compare equal.
    """
    if not positions:
        return ()
    out = []
    previous_root = None
    for frets in positions:
        if not frets:
            continue
        root = min(frets)
        shape = tuple(sorted(fret - root for fret in frets))
        move = 0 if previous_root is None else root - previous_root
        out.append((shape, move))
        previous_root = root
    return tuple(out)


def is_chorded(signature: tuple) -> bool:
    """Does this figure contain a chord at all?"""
    return any(len(shape) > 1 for shape, _ in signature)
