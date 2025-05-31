import pandas as pd
from pathlib import Path

DROP_COLS = {
    "accelerometer": [
        'acc_x_skewness', 'acc_x_kurtosis',
        'acc_y_skewness', 'acc_y_kurtosis',
        'acc_z_skewness', 'acc_z_kurtosis',
        'magnitude_skewness', 'magnitude_kurtosis'
    ],
    "gyroscope": [
        'gyro_x_skewness', 'gyro_x_kurtosis',
        'gyro_y_skewness', 'gyro_y_kurtosis',
        'gyro_z_skewness', 'gyro_z_kurtosis',
        'magnitude_skewness', 'magnitude_kurtosis'
    ]
}

ID_COLS = ["subject_id", "session_number", "start_ms", "end_ms"]

rename_cols = ['corr_xz', 'corr_yz','magnitude_mean', 
            'magnitude_std', 'magnitude_min', 'magnitude_max']


def load_modality(file_path: Path, modality: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    df = df.drop(columns=DROP_COLS[modality], errors='ignore')

    prefix = 'acc_' if modality == 'accelerometer' else 'gyro_'
    rename_map = {c: f"{prefix}{c}" for c in rename_cols}
    df = df.rename(columns=rename_map)

    return df


def fuse_modalities(acc_path: Path, gyro_path: Path) -> pd.DataFrame:
    df_acc = load_modality(acc_path, 'accelerometer')
    df_gyro = load_modality(gyro_path, 'gyroscope')

    df_merged = pd.merge(
        df_acc,
        df_gyro,
        on=ID_COLS,
        how='inner',
        validate='one_to_one'
    )
    return df_merged


if __name__ == '__main__':
    base_path = Path("data/features/hmog_None")
    acc_file = base_path / "accelerometer_features.parquet"
    gyro_file = base_path / "gyroscope_features.parquet"

    fused_df = fuse_modalities(acc_file, gyro_file)
    print(f"Fusão concluída: {len(fused_df)} janelas unidas.")

    fused_df.to_parquet(base_path / "multimodal_features.parquet", index=False)
    print("Salvo em features_multimodal.parquet")
