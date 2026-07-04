from collections import deque, defaultdict

import pandas as pd


class HeuristicLearner:
    """Base class for heuristic speculative-flow selectors.

    Maintains a sliding window of the last ``window_size`` LTIs' per-flow hit
    counts. Each LTI the simulation calls :meth:`observe_lti` with the hits
    generated during the just-finished interval, then :meth:`rank_flows` to get
    a best-first ordering of controller flows to install speculatively.

    Subclasses only need to implement :meth:`rank_flows`. New heuristics can be
    added by subclassing and registering in :func:`build_heuristic_learner`.
    """

    # Provided for interface compatibility with the RL learners (unused here).
    num_states = 0

    def __init__(self, args):
        self.args = args
        self.window_size = int(getattr(args, 'speculative_window_size', 100))
        self.window = deque(maxlen=self.window_size)

    def observe_lti(self, lti_hits):
        """Push the just-finished LTI's ``{(Source, Destination): hits}`` map."""
        self.window.append(dict(lti_hits))

    def rank_flows(self, controller_table):
        """Return a DataFrame of ``[Source, Destination]`` ranked best-first."""
        raise NotImplementedError

    def get_info(self):
        return {
            'heuristic': 'base',
            'window_size': self.window_size,
        }


class HitCountHeuristicLearner(HeuristicLearner):
    """Rank flows by total hit count summed across the sliding window."""

    def _aggregate_window_hits(self):
        totals = defaultdict(int)
        for lti_hits in self.window:
            for flow_key, hits in lti_hits.items():
                totals[flow_key] += hits
        return totals

    def rank_flows(self, controller_table):
        totals = self._aggregate_window_hits()
        if not totals:
            return controller_table.iloc[0:0][['Source', 'Destination']].copy()

        ranked = pd.DataFrame(
            [(src, dst, hits) for (src, dst), hits in totals.items()],
            columns=['Source', 'Destination', 'windowed_hits'],
        )
        ranked = ranked.sort_values(by='windowed_hits', ascending=False)

        # Keep only flows that currently exist in the controller table.
        valid = controller_table[['Source', 'Destination']]
        ranked = ranked.merge(valid, on=['Source', 'Destination'], how='inner')
        return ranked[['Source', 'Destination']].reset_index(drop=True)

    def get_info(self):
        info = super().get_info()
        info['heuristic'] = 'hitcount'
        return info


def build_heuristic_learner(args):
    """Construct the heuristic selector chosen by ``--heuristic``."""
    heuristic = getattr(args, 'heuristic', 'hitcount')
    if heuristic == 'hitcount':
        return HitCountHeuristicLearner(args)
    raise ValueError(f"Unknown heuristic '{heuristic}'")
