"""
Preprocessing functions for ToF data
"""

import numpy as np
import torch
from typing import Tuple


def normalize_ir_by_distance(
    ir: np.ndarray,
    depth: np.ndarray,
    min_depth: float = 0.5
) -> np.ndarray:
    """
    距離逆二乗則によるIR強度の補正
    
    I_normalized = I_raw * d^2
    
    これにより、距離に依存しない材質固有の反射率（アルベド）を推定可能
    
    Args:
        ir: IR強度 (H, W)
        depth: 距離マップ (H, W)
        min_depth: 最小距離（ゼロ除算防止）
    
    Returns:
        正規化されたIR強度
    """
    depth_safe = np.maximum(depth, min_depth)
    ir_normalized = ir * (depth_safe ** 2)
    return ir_normalized


def compute_surface_normals(
    depth: np.ndarray,
    focal_length: float = 525.0,
    pixel_size: float = 1.0
) -> np.ndarray:
    """
    深度マップから表面法線を計算
    
    Args:
        depth: 距離マップ (H, W)
        focal_length: 焦点距離（ピクセル単位）
        pixel_size: ピクセルサイズ
    
    Returns:
        法線マップ (3, H, W), 各チャネルは (nx, ny, nz)
    """
    h, w = depth.shape
    
    # 勾配を計算
    dz_dx = np.zeros_like(depth)
    dz_dy = np.zeros_like(depth)
    
    # 中心差分
    dz_dx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) / 2
    dz_dy[1:-1, :] = (depth[2:, :] - depth[:-2, :]) / 2
    
    # 端のピクセル
    dz_dx[:, 0] = depth[:, 1] - depth[:, 0]
    dz_dx[:, -1] = depth[:, -1] - depth[:, -2]
    dz_dy[0, :] = depth[1, :] - depth[0, :]
    dz_dy[-1, :] = depth[-1, :] - depth[-2, :]
    
    # 法線ベクトル (右手座標系)
    nx = -dz_dx * focal_length
    ny = -dz_dy * focal_length
    nz = np.ones_like(depth) * pixel_size
    
    # 正規化
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    norm = np.maximum(norm, 1e-10)  # ゼロ除算防止
    
    normals = np.stack([
        nx / norm,
        ny / norm,
        nz / norm
    ], axis=0)
    
    return normals.astype(np.float32)


def compute_curvature(
    depth: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:
    """
    深度マップから曲率を計算
    
    Args:
        depth: 距離マップ (H, W)
        kernel_size: 局所ウィンドウサイズ
    
    Returns:
        曲率マップ (H, W)
    """
    # Laplacian (second derivative) で曲率を近似
    laplacian = np.zeros_like(depth)
    
    laplacian[1:-1, 1:-1] = (
        depth[:-2, 1:-1] + depth[2:, 1:-1] +
        depth[1:-1, :-2] + depth[1:-1, 2:] -
        4 * depth[1:-1, 1:-1]
    )
    
    return laplacian.astype(np.float32)


def extract_raw_features(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Raw相関データ (Q1-Q4) から物理量を抽出
    
    ToFの測定原理:
    Q1 = B + A*cos(φ)
    Q2 = B + A*cos(φ + π/2) = B - A*sin(φ)
    Q3 = B + A*cos(φ + π) = B - A*cos(φ)
    Q4 = B + A*cos(φ + 3π/2) = B + A*sin(φ)
    
    From these:
    φ = atan2(Q4 - Q2, Q1 - Q3)
    A = sqrt((Q1-Q3)^2 + (Q4-Q2)^2) / 2
    B = (Q1 + Q2 + Q3 + Q4) / 4
    
    Args:
        raw: Raw相関データ (4, H, W)
    
    Returns:
        phase: 位相 (H, W)
        amplitude: 振幅 (H, W)
        offset: オフセット (H, W)
    """
    q1, q2, q3, q4 = raw[0], raw[1], raw[2], raw[3]
    
    # 位相
    phase = np.arctan2(q4 - q2, q1 - q3)
    
    # 振幅
    amplitude = np.sqrt((q1 - q3)**2 + (q4 - q2)**2) / 2
    
    # オフセット（環境光）
    offset = (q1 + q2 + q3 + q4) / 4
    
    return phase.astype(np.float32), amplitude.astype(np.float32), offset.astype(np.float32)


def apply_incidence_angle_correction(
    ir: np.ndarray,
    normals: np.ndarray,
    view_direction: np.ndarray = None
) -> np.ndarray:
    """
    入射角によるIR強度の補正（Lambert反射を仮定）
    
    I_corrected = I_raw / cos(θ)
    where cos(θ) = n・v
    
    Args:
        ir: IR強度 (H, W)
        normals: 法線マップ (3, H, W)
        view_direction: 視線方向（省略時は z軸方向 [0, 0, 1]）
    
    Returns:
        入射角補正されたIR強度
    """
    if view_direction is None:
        view_direction = np.array([0, 0, 1])
    
    # cos(θ) = n・v
    cos_theta = (
        normals[0] * view_direction[0] +
        normals[1] * view_direction[1] +
        normals[2] * view_direction[2]
    )
    
    # 負の値（裏面）や極端な角度をクリップ
    cos_theta = np.clip(cos_theta, 0.1, 1.0)
    
    ir_corrected = ir / cos_theta
    
    return ir_corrected.astype(np.float32)


class ToFPreprocessor:
    """
    ToFデータの前処理パイプライン
    """
    
    def __init__(
        self,
        normalize_depth: bool = True,
        correct_ir_distance: bool = True,
        correct_ir_angle: bool = False,
        compute_normals: bool = True,
        extract_raw_physics: bool = False
    ):
        self.normalize_depth = normalize_depth
        self.correct_ir_distance = correct_ir_distance
        self.correct_ir_angle = correct_ir_angle
        self.compute_normals = compute_normals
        self.extract_raw_physics = extract_raw_physics
    
    def __call__(
        self,
        depth: np.ndarray,
        ir: np.ndarray,
        raw: np.ndarray
    ) -> dict:
        """
        前処理を適用
        
        Args:
            depth: 距離マップ (H, W)
            ir: IR強度 (H, W)
            raw: Raw相関データ (4, H, W)
        
        Returns:
            処理済みデータの辞書
        """
        result = {
            'depth': depth.copy(),
            'ir': ir.copy(),
            'raw': raw.copy()
        }
        
        # 法線計算
        if self.compute_normals:
            normals = compute_surface_normals(depth)
            result['normals'] = normals
        
        # 距離正規化
        if self.normalize_depth:
            result['depth'] = (result['depth'] - 0.5) / 3.5  # [0.5, 4.0] -> [0, 1]
        
        # IR距離補正
        if self.correct_ir_distance:
            result['ir'] = normalize_ir_by_distance(result['ir'], depth)
        
        # IR入射角補正
        if self.correct_ir_angle and 'normals' in result:
            result['ir'] = apply_incidence_angle_correction(
                result['ir'], result['normals']
            )
        
        # Raw物理量抽出
        if self.extract_raw_physics:
            phase, amplitude, offset = extract_raw_features(raw)
            result['raw_phase'] = phase
            result['raw_amplitude'] = amplitude
            result['raw_offset'] = offset
        
        return result
