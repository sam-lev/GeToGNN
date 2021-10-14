
# Third party imports
import numpy as np



cached_msc_path = None






def is_ridge_arc(arc, msc):
    return 0 not in [
        msc.nodes[arc.node_ids[0]].index,
        msc.nodes[arc.node_ids[1]].index,
    ]


def is_valley_arc(arc, msc):
    return 2 not in [
        msc.nodes[arc.node_ids[0]].index,
        msc.nodes[arc.node_ids[1]].index,
    ]

