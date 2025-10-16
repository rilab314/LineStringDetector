from Figure import Figure_1, Figure_2, Figure_5
from Table import Table_1, Table_2
from visualization import visualize_coco_matches, visualize_sample_image
import config as cfg

try:
    from tqdm import tqdm
    def progress(it, **kw): return tqdm(it, **kw)
except Exception:
    def progress(it, **kw): return it

def main():
    for thickness in progress(cfg.THICKNESS, desc="thickness"):
        for sample_stride in progress(cfg.SAMPLE_STRIDE, desc=f"stride (t={thickness})"):
            cfg.thickness = thickness
            cfg.sample_stride = sample_stride
            cfg.update_paths()

            print("\n" + "=" * 80)
            print(f"[RUN] thickness={cfg.thickness}, sample_stride={cfg.sample_stride}")
            print(f"[RESULT_PATH] {cfg.RESULT_PATH}")
            print("=" * 80)

            Figure_1.main()
            Figure_2.main()
            Figure_5.main()
            Table_1.main()
            Table_2.main()
            visualize_coco_matches.main()
            visualize_sample_image.main()

if __name__ == '__main__':
    main()
