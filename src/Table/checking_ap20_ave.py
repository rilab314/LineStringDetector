import pandas as pd
import numpy as np
import os

# --- 경로 설정 ---
table_folder = '/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Archive/Dataset/unzips/LaneDetector(copy)/test/result/Table'
TABLE1_CSV = os.path.join(table_folder, 'table_1.csv')
TABLE2_CSV = os.path.join(table_folder, 'table_2.csv')


def verify():
    if not os.path.exists(TABLE1_CSV) or not os.path.exists(TABLE2_CSV):
        print("CSV 파일을 찾을 수 없습니다.")
        return
    t1 = pd.read_csv(TABLE1_CSV)
    t2 = pd.read_csv(TABLE2_CSV)

    t2_vals = t2['AP20'].values
    t2_avg = np.mean(t2_vals)
    t1_ap20 = pd.to_numeric(t1.loc[t1.iloc[:, 0].str.contains('merged', na=False), 'AP20']).values[0]

    print("=" * 60)
    print(f"Table 2 AP20 합계: {np.sum(t2_vals):.6f} (대상 클래스: 8개)")
    print(f"Table 2 산술 평균: {t2_avg:.6f}")
    print(f"Table 1 기록 AP20: {t1_ap20:.6f}")
    print("-" * 60)
    if abs(t2_avg - t1_ap20) < 1e-5:
        print("✅ 결과: Table 1과 2의 수치가 일치합니다.")
    else:
        print(f"❌ 결과: 수치 차이 발생 (Diff: {abs(t2_avg - t1_ap20):.6f})")
    print("=" * 60)


if __name__ == "__main__":
    verify()