import itertools
import math
from chart.time_conversion import convert_notes_to_seconds


class SimpleTokenizerGuitar():
    def __init__(self, exclude_open_chords=True):

        self.exclude_open_chords = exclude_open_chords
        # Define a mapping between pressed lanes (5 fret + open) and note index
        # If exclude_open_chords is True, exclude every chord with 7 except 7 alone

        indices = [i for i in range(5)]
        if not exclude_open_chords:
            indices = indices + [7]
        all_combinations = []
        for r in range(1, len(indices) + 1):
            combos = list(itertools.combinations(indices, r))
            all_combinations.extend(combos)

        self.mapping_noteseqs2int = {v:idx for idx, v in enumerate(all_combinations)}
        if self.exclude_open_chords:
            self.mapping_noteseqs2int[(7,)] = len(all_combinations)
        self.reverse_map = {v: k for k, v in self.mapping_noteseqs2int.items()}


    def _normalize_lanes(self, lanes):
        lanes = sorted(set(lanes))
        if self.exclude_open_chords and len(lanes) > 1 and lanes[-1] == 7:
            lanes.pop()
        return lanes

    def _append_encoded_group(self, output, tick, lanes, duration, is5, is6, is_power):
        lanes = self._normalize_lanes(lanes)
        if not lanes:
            return
        mapped = self.mapping_noteseqs2int.get(tuple(lanes))
        if mapped is None:
            raise ValueError(f"Unknown note sequence {lanes} at tick {tick}")
        output.append((tick, mapped, duration, {
            'is5': is5, 'is6': is6, 'isS': is_power,
        }))

    @staticmethod
    def _power_intervals(note_list):
        return sorted(
            (tick, tick + duration)
            for tick, note_type, _, duration in note_list
            if note_type == 'S'
        )

    def encode(self, note_list):
        encoded_notes = []
        last_tick = None
        seq_notes = []
        last_duration = None

        has_is5 = False
        has_is6 = False

        # Extract star power intervals from 'S' notes
        power_intervals = self._power_intervals(note_list)
        power_index = 0

        def is_in_power(tick):
            nonlocal power_index
            while power_index < len(power_intervals) and power_intervals[power_index][1] <= tick:
                power_index += 1
            return (power_index < len(power_intervals)
                    and power_intervals[power_index][0] <= tick < power_intervals[power_index][1])

        for tick, note_type, lane, duration in note_list:
            if note_type == 'S':
                continue  # star powers already handled

            # If we changed tick, output previous group first
            if last_tick is not None and tick != last_tick:
                # Clone Hero charts may contain duplicate lanes at one tick.
                self._append_encoded_group(encoded_notes, last_tick, seq_notes,
                                           last_duration, has_is5, has_is6,
                                           is_in_power(last_tick))

                # reset for new tick
                seq_notes = []
                has_is5 = False
                has_is6 = False
                last_duration = None

            # Process lane 5 or 6 notes: mark flag but do not add lane to seq_notes
            if lane == 5:
                has_is5 = True
            elif lane == 6:
                has_is6 = True
            else:
                seq_notes.append(lane)
                # Update duration to max of all lanes for this tick
                if last_duration is None:
                    last_duration = duration
                else:
                    last_duration = max(last_duration, duration)

            last_tick = tick

        # Flush last group
        if last_tick is not None and seq_notes:
            # Clone Hero charts may contain duplicate lanes at one tick.
            self._append_encoded_group(encoded_notes, last_tick, seq_notes,
                                       last_duration, has_is5, has_is6,
                                       is_in_power(last_tick))

        return encoded_notes
    
    def decode(self, encoded_notes):
        note_list = []

        # Step 1: Precompute star power (isS) intervals
        power_start_ticks = {}  # tick -> duration

        in_power = False
        sustain_start = None

        for tick, _, _, attrs in encoded_notes:
            if attrs.get('isS', False):
                if not in_power:
                    in_power = True
                    sustain_start = tick
            else:
                if in_power:
                    power_start_ticks[sustain_start] = tick - sustain_start
                    in_power = False

        if in_power:
            last_tick = encoded_notes[-1][0]
            last_duration = encoded_notes[-1][2]
            power_start_ticks[sustain_start] = last_tick + last_duration - sustain_start

        # Step 2: Decode notes inline, insert S immediately after first isS
        for tick, mapped, duration, attrs in encoded_notes:
            lanes = self.reverse_map[mapped]

            for lane in lanes:
                note_list.append((tick, 'N', lane, duration))

            #sustain is not assigned to those notes
            if attrs.get('is5', False):
                note_list.append((tick, 'N', 5, 0))
            if attrs.get('is6', False):
                note_list.append((tick, 'N', 6, 0))

            # Add S only when this tick is the start of a sustain
            if tick in power_start_ticks:
                sustain_duration = power_start_ticks[tick]
                note_list.append((tick, 'S', 2, sustain_duration))  # Use lane 2 or another if desired

        # Already in order, no need to sort
        return note_list  

    def format_seconds(self, notes, bpm_events, resolution=192, offset=0):
        return convert_notes_to_seconds(notes, bpm_events, resolution, offset)


    def discretize_time(self, time_list, tokens_list, pad_token_id, grid_ms, window_seconds, start_time=0.0):
        """
        Map tokens to a time-grid, discretized relative to a given start time.
        
        Parameters:
        - time_list: List of float timestamps (in seconds) for each token.
        - tokens_list: List of corresponding tokens.
        - pad_token_id: Integer ID used for padding empty grid slots.
        - grid_ms: Grid resolution in milliseconds (e.g., 10 for 10ms bins).
        - window_seconds: Total window duration in seconds to cover with the grid.
        - start_time: Float start timestamp (in seconds); times are relative to this.
        
        The function computes relative times as (t - start_time), rounds them to the nearest
        grid step, and places tokens on the grid. Times before start_time or beyond the window
        are ignored/clipped as needed.
        """

        if not isinstance(pad_token_id, int):
            raise TypeError("pad_token_id must be an integer")
        if grid_ms <= 0 or window_seconds <= 0:
            raise ValueError("grid_ms and window_seconds must be positive")
        if len(tokens_list) != len(time_list):
            raise ValueError("tokens and times_sec must have the same length")

        grid_s = grid_ms / 1000.0
        steps = window_seconds / grid_s
        if not math.isclose(steps, round(steps), abs_tol=1e-9):
            raise ValueError("window_seconds must be divisible by the time grid")
        
        # Compute relative times and check min delta (among all pairs, for collision safety)
        # It could be a time window with no tokens
        rel_times = [t - start_time for t in time_list]
        if len(rel_times) > 1:
            min_dt = min_delta(time_list)
            if min_dt < grid_s: 
                raise ValueError("Min dt too short will cause collision in discretization")

        n_steps = int(window_seconds / grid_s)  # Total bins in the window
        grid = [pad_token_id] * n_steps
         
        # Round each relative time to nearest grid step and place token if in bounds
        for token, rel_t in zip(tokens_list, rel_times):
            if rel_t < 0:
                continue
            idx = int(round(rel_t / grid_s))
            if 0 <= idx < len(grid): #with < a note exactly at rel_time=window_seconds is excluded
                grid[idx] = token

        return grid



def min_delta(times_sec):
    times_sorted = sorted(times_sec)
    return min(
        times_sorted[i] - times_sorted[i - 1]
        for i in range(1, len(times_sorted))
    )
