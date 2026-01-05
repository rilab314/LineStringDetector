from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class SatelliteDataset(CustomDataset):
    RILAB_LG_12 = [
        {'id': 0, 'name': 'ignore', 'color': (0, 0, 0)},
        {'id': 1, 'name': 'center_line', 'color': (77, 77, 255)},
        {'id': 2, 'name': 'u_turn_zone_line', 'color': (77, 178, 255)},
        {'id': 3, 'name': 'lane_line', 'color': (77, 255, 77)},
        {'id': 4, 'name': 'bus_only_lane', 'color': (255, 153, 77)},
        {'id': 5, 'name': 'edge_line', 'color': (255, 77, 77)},
        {'id': 6, 'name': 'path_change_restriction_line', 'color': (178, 77, 255)},
        {'id': 7, 'name': 'no_parking_stopping_line', 'color': (77, 255, 178)},
        {'id': 8, 'name': 'guiding_line', 'color': (255, 178, 77)},
        {'id': 9, 'name': 'stop_line', 'color': (77, 102, 255)},
        {'id': 10, 'name': 'safety_zone', 'color': (255, 77, 128)},
        {'id': 11, 'name': 'bicycle_lane', 'color': (128, 255, 77)},
    ]

    CLASSES = [item['name'] for item in RILAB_LG_12]
    PALETTE = [item['color'] for item in RILAB_LG_12]

    def __init__(self, classes=None, palette=None, **kwargs):
        self.CLASSES = classes if classes is not None else self.CLASSES
        self.PALETTE = palette if palette is not None else self.PALETTE

        super(SatelliteDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs
        )
