import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

import torch
import pandas as pd


CHECKPOINTS = {
    "internimage_large": f"{cfg.DATA_ROOT}/Internimage/checkpoint/upernet_internimage_l_512_160k_satellite/best_mIoU_iter_160000.pth",
    "mask2former_small": f"{cfg.DATA_ROOT}/mask2former/checkpoint/mask2former_swin-s_8xb2-160k_ade20k-512x512/iter_160000.pth",
    "mask2former_large": f"{cfg.DATA_ROOT}/mask2former/checkpoint/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640/iter_160000.pth",
}


def main():
    rows = []
    for model_name, pth_path in CHECKPOINTS.items():
        print(f"[{model_name}] 파라미터 계산 중: {pth_path}")
        total_params = count_params(pth_path)
        total_m = total_params / 1e6
        print(f"  -> 전체 파라미터 수: {total_params:,} ({total_m:.1f}M)")
        rows.append({"model": model_name, "total_params": total_params, "total_params_M": f"{total_m:.1f}"})

    out_path = os.path.join(cfg.RESULT_PATH, "num_params.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nCSV 저장 완료: {out_path}")


def count_params(pth_path: str) -> int:
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    # state_dict 추출 (중첩 구조 대응)
    if isinstance(ckpt, dict):
        state_dict = (
            ckpt.get("state_dict")
            or ckpt.get("model")
            or ckpt.get("model_state_dict")
            or ckpt
        )
    else:
        state_dict = ckpt
    return sum(t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor))


if __name__ == "__main__":
    main()
