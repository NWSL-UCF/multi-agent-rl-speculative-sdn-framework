"""Helpers for limiting packet replay to a trace time window."""


def get_trace_start_time(args):
    return float(getattr(args, "trace_start_time", 0.0))


def get_simulation_end_time(args):
    return get_trace_start_time(args) + float(args.simulation_time)


def get_replay_start_time(value):
    return float(value.iloc[0].iloc[0])


def should_stop_simulation(args, value, packet_counter):
    """Return True when replay has reached the end of the simulation window."""
    if packet_counter >= len(value) - 1:
        return True

    current_time = float(value.iloc[packet_counter].iloc[0])
    return current_time > get_simulation_end_time(args)
