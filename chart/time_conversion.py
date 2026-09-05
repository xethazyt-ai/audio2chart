import bisect

def preprocess_bpm_segments(bpm_events, resolution):
    """
    Preprocess BPM events into segments with cumulative time tracking
    
    Args:
        bpm_events: List of (tick, bpm_value) tuples where bpm_value is multiplied by 1000
        resolution: Ticks per quarter note (PPQ)
    
    Returns:
        List of (tick, bpm, cumulative_time) tuples
    """
    if not bpm_events:
        raise ValueError("bpm_events cannot be empty")
    
    # Charts are not required to order synchronization events.
    bpm_events = sorted(bpm_events)
    
    segments = []
    last_tick = 0
    cumulative_time = 0.0
    
    for tick, bpm_raw in bpm_events:
        # Convert BPM from raw value (divided by 1000)
        bpm = bpm_raw / 1000.0
        
        if bpm <= 0:
            raise ValueError(f"Invalid BPM value: {bpm}")
        
        # Calculate time elapsed since last BPM change
        if segments:  # Not the first segment
            prev_bpm = segments[-1][1]
            delta_ticks = tick - last_tick
            # Time = (ticks / resolution) * (60 / bpm) 
            # This gives us the time in seconds for the tick duration
            time_elapsed = (delta_ticks / resolution) * (60.0 / prev_bpm)
            cumulative_time += time_elapsed
        
        segments.append((tick, bpm, cumulative_time))
        last_tick = tick
    
    return segments


def tick_to_seconds(tick, bmp_segments, resolution, segment_ticks=None):
    """
    Convert a tick position to absolute time in seconds
    
    Args:
        tick: The tick position to convert
        bmp_segments: Preprocessed BMP segments from preprocess_bpm_segments()
        resolution: Ticks per quarter note (PPQ)
    
    Returns:
        Absolute time in seconds
    """
    if not bmp_segments:
        raise ValueError("bmp_segments cannot be empty")
    
    # Find the appropriate BPM segment
    ticks = segment_ticks if segment_ticks is not None else [seg[0] for seg in bmp_segments]
    idx = bisect.bisect_right(ticks, tick) - 1
    
    # Handle edge case where tick is before first BPM event
    if idx < 0:
        idx = 0
        # If tick is before first BPM event, use first BPM for the entire duration
        base_tick, bpm, base_time = bmp_segments[0]
        delta_ticks = tick - 0  # From start
        delta_time = (delta_ticks / resolution) * (60.0 / bpm)
        return delta_time
    
    base_tick, bpm, base_time = bmp_segments[idx]
    delta_ticks = tick - base_tick
    delta_time = (delta_ticks / resolution) * (60.0 / bpm)
    
    return base_time + delta_time


def convert_notes_to_seconds(notes, bpm_events, resolution, offset=0.0):
    """
    Convert notes from tick-based timing to absolute time in seconds
    
    Args:
        notes: List of (tick, lane, sustain_ticks, attr) tuples
        bpm_events: List of (tick, bpm_raw) tuples where bpm_raw is * 1000
        resolution: Ticks per quarter note (PPQ)
        offset: Initial time offset in seconds (default: 0.0)
    
    Returns:
        List of (absolute_time, lane, duration, attr) tuples
    """
    if not notes:
        raise ValueError("notes cannot be empty")
    
    if not bpm_events:
        raise ValueError("bpm_events cannot be empty")
    
    if 'N' in notes[0] or 'S' in notes[0]:
        raise ValueError("" \
        "Invalid input format for conversion, input only encoded seqs in format tick,note_idx,sustain,attrs."
    )
    
    bmp_segments = preprocess_bpm_segments(bpm_events, resolution)
    segment_ticks = [segment[0] for segment in bmp_segments]
    note_times = []
    
    for tick, note_idx, sustain, attr in notes:
        # Convert start tick to absolute time
        start_sec = tick_to_seconds(tick, bmp_segments, resolution, segment_ticks)
        
        # Convert sustain duration to time
        if sustain > 0:
            end_sec = tick_to_seconds(tick + sustain, bmp_segments, resolution, segment_ticks)
            duration = end_sec - start_sec
        else:
            duration = 0.0
        
        # Add offset to get final absolute time
        absolute_time = offset + start_sec
        note_times.append((absolute_time, note_idx, duration, attr))
    
    return note_times

def time_to_tick(time_sec, bpm, resolution):
    return int(round(time_sec * bpm / 60.0 * resolution))

def seconds_to_tick(time_sec, bmp_segments, resolution, segment_times=None):
    """
    Convert an absolute time in seconds to a tick position

    Inverse of tick_to_seconds(), so it honours a piecewise-constant tempo map
    instead of assuming a single BPM for the whole song.

    Args:
        time_sec: The absolute time to convert
        bmp_segments: Preprocessed BMP segments from preprocess_bpm_segments()
        resolution: Ticks per quarter note (PPQ)

    Returns:
        Tick position
    """
    if not bmp_segments:
        raise ValueError("bmp_segments cannot be empty")

    times = segment_times if segment_times is not None else [seg[2] for seg in bmp_segments]
    idx = bisect.bisect_right(times, time_sec) - 1

    if idx < 0:
        idx = 0

    base_tick, bpm, base_time = bmp_segments[idx]
    delta_ticks = (time_sec - base_time) * (bpm / 60.0) * resolution

    return int(round(base_tick + delta_ticks))


def convert_notes_to_ticks(tokens_list, time_list, fixed_bpm=200, resolution=480,
                           bpm_events=None, snap=0, pad_token_id=34, tokenizer=None):
    """
    Place decoded tokens on the tick grid.

    Passing bpm_events uses that tempo map instead of fixed_bpm; the caller must
    write the same map into [SyncTrack] or the notes will not line up with the
    audio. snap rounds each note to the nearest 1/snap note (e.g. snap=32 for
    thirty-second notes), which is only meaningful once the grid is correct.
    """

    attrs = {
        'is5': False,
        'is6': False,
        'isS': False,
    }

    if bpm_events:
        segments = preprocess_bpm_segments(bpm_events, resolution)
        segment_times = [segment[2] for segment in segments]

        def to_tick(t):
            return seconds_to_tick(t, segments, resolution, segment_times)
    else:
        def to_tick(t):
            return time_to_tick(t, fixed_bpm, resolution)

    grid = int(round(resolution * 4 / snap)) if snap else 0

    tick_notes = []
    seen = set()

    for t, note in zip(time_list, tokens_list):
        if note == pad_token_id:
            continue
        if tokenizer is not None and not tokenizer.is_note_token(note):
            continue  # <bos>/<eos> or out of range
        tick = to_tick(t)
        if grid:
            tick = int(round(tick / grid)) * grid
            # snapping can collapse neighbouring tokens onto the same lane
            if (tick, note) in seen:
                continue
            seen.add((tick, note))
        if tokenizer is not None:
            note_duration = tokenizer.sustain_ticks(note, resolution)
            note_attrs = tokenizer.attrs_of(note)
        else:
            note_duration, note_attrs = 0, attrs
        tick_notes.append((tick, note, note_duration, note_attrs))

    return tick_notes
