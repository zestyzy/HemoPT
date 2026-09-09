from __future__ import annotations
import math
from typing import Callable, Dict, Optional
import torch

# ================================================================
# Coordinate conventions follow each dataset-specific preprocessing script.
# ================================================================

def _direction_craft(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    # x: (b, n, 3)
    # cond: (b, 1, 3) -> [mach, aoa, beta]
    b, n, _ = x.shape
    device = x.device
    dtype = x.dtype

    aoa = cond[:, :, 1:2]  # (b,1,1) degrees
    beta = cond[:, :, 2:3]  # (b,1,1) degrees
    mach = cond[:, :, 0:1]  # (b,1,1)

    vx = torch.cos(torch.pi * aoa / 180.0).repeat(1, n, 1) * torch.cos(torch.pi * beta / 180.0).repeat(1, n, 1)
    vy = torch.sin(torch.pi * aoa / 180.0).repeat(1, n, 1)
    vz = torch.cos(torch.pi * aoa / 180.0).repeat(1, n, 1) * torch.sin(torch.pi * beta / 180.0).repeat(1, n, 1)

    v = torch.cat([vx, vy, vz], dim=-1).to(device=device, dtype=dtype)
    extra = (mach.repeat(1, n, 1) / 3.0).to(device=device, dtype=dtype)
    return torch.cat([v, extra], dim=-1)


def _direction_nasa(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    # cond: (b,1,2) -> [mach, aoa]  or (b,1,>=3) if you append others
    b, n, _ = x.shape
    device, dtype = x.device, x.dtype

    aoa = cond[:, :, 1:2]
    mach = cond[:, :, 0:1]

    vx = torch.cos(torch.pi * aoa / 180.0).repeat(1, n, 1)
    vy = torch.sin(torch.pi * aoa / 180.0).repeat(1, n, 1)
    vz = torch.zeros(b, n, 1, device=device, dtype=dtype)

    v = torch.cat([vx, vy, vz], dim=-1)
    extra = (mach.repeat(1, n, 1) * 1.6).to(dtype)
    return torch.cat([v, extra], dim=-1)


def _direction_crash(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    # cond: (b,1,1) or (b,1,*) -> here assume first dim is angle in radians
    b, n, _ = x.shape
    device, dtype = x.device, x.dtype

    angle = cond[..., :1]  # (b,1,1)
    vx = torch.cos(torch.pi * angle / 180.0).repeat(1, n, 1)
    vy = torch.zeros(b, n, 1, device=device, dtype=dtype)
    vz = torch.sin(torch.pi * angle / 180.0).repeat(1, n, 1)
    v = torch.cat([vx, vy, vz], dim=-1).to(dtype)

    x_max = torch.max(x[:, :, 0:1], dim=1, keepdim=True)[0]
    x_min = torch.min(x[:, :, 0:1], dim=1, keepdim=True)[0]
    speed = (x[:, :, 0:1] - x_min) / (x_max - x_min + 1e-8)
    extra = (speed * 0.5).to(dtype)
    return torch.cat([v, extra], dim=-1)


def _direction_hull(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    # cond: (b,1,1) -> [angle degrees]
    b, n, _ = x.shape
    device, dtype = x.device, x.dtype

    angle = cond  # degrees
    vx = torch.cos(torch.pi * angle / 180.0).repeat(1, n, 1)
    vy = torch.zeros(b, n, 1, device=device, dtype=dtype)
    vz = torch.sin(torch.pi * angle / 180.0).repeat(1, n, 1)
    v = torch.cat([vx, vy, vz], dim=-1).to(dtype)

    thr = 0.17428
    mask = (x[:, :, 1] > thr).unsqueeze(-1)  # True means set extra=0
    extra = (0.3 * (~mask).to(dtype)).to(device=device)  # (b,n,1)
    return torch.cat([v, extra], dim=-1)


def _direction_drivAerML(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    # cond: (b,1,2) -> [weight, angle_degrees]
    b, n, _ = x.shape
    device, dtype = x.device, x.dtype

    speed = 0.3 # default
    vx = torch.ones(b, n, 1, device=device, dtype=dtype)
    vy = torch.zeros(b, n, 1, device=device, dtype=dtype)
    vz = torch.zeros(b, n, 1, device=device, dtype=dtype)
    w = torch.ones(b, n, 1, device=device, dtype=dtype) * speed

    return torch.cat([vx, vy, vz, w], dim=-1).to(dtype)


def _direction_hemo(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """
    Dynamics direction for intravascular blood flow.
    cond: (b, 1, 3) -> [vessel_type, inlet_velocity, viscosity]

    Uses a simplified Poiseuille-like velocity profile:
      - Primary flow along the vessel's longitudinal axis (estimated as the
        direction of maximum point spread)
      - Parabolic profile: speed varies with distance from center
    """
    b, n, _ = x.shape
    device, dtype = x.device, x.dtype

    inlet_vel = cond[:, :, 1:2]  # (b, 1, 1)

    # Estimate vessel axis direction from point cloud spread
    x_coords = x[:, :, 0:1]  # (b, n, 1)
    y_coords = x[:, :, 1:2]
    z_coords = x[:, :, 2:3]

    # Variance along each axis -> axis with max spread = flow direction
    var_x = torch.var(x_coords, dim=1, keepdim=True)
    var_y = torch.var(y_coords, dim=1, keepdim=True)
    var_z = torch.var(z_coords, dim=1, keepdim=True)

    # Flow direction: unit vector along axis of maximum spread
    vx = (var_x >= var_y) & (var_x >= var_z)
    vy = (var_y > var_x) & (var_y >= var_z)
    vz = ~vx & ~vy

    dir_x = vx.float() * torch.sign(x_coords.max(dim=1, keepdim=True)[0] - x_coords.min(dim=1, keepdim=True)[0])
    dir_y = vy.float() * torch.sign(y_coords.max(dim=1, keepdim=True)[0] - y_coords.min(dim=1, keepdim=True)[0])
    dir_z = vz.float() * torch.sign(z_coords.max(dim=1, keepdim=True)[0] - z_coords.min(dim=1, keepdim=True)[0])

    direction = torch.cat([dir_x, dir_y, dir_z], dim=-1)  # (b, 1, 3)
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

    # Parabolic speed profile: max at center, zero at wall
    # Approximate radial distance from centerline
    center = x.mean(dim=1, keepdim=True)  # (b, 1, 3)
    diff = x - center  # (b, n, 3)

    # Project out the axial component -> radial distance
    axial = (diff * direction).sum(dim=-1, keepdim=True)  # (b, n, 1)
    radial_vec = diff - axial * direction
    r = radial_vec.norm(dim=-1, keepdim=True)  # (b, n, 1)

    r_max = r.max(dim=1, keepdim=True)[0] + 1e-8  # (b, 1, 1)
    r_norm = r / r_max  # (b, n, 1) in [0, 1]

    # Poiseuille: v(r) = v_max * (1 - (r/R)^2)
    speed = inlet_vel.repeat(1, n, 1) * (1.0 - r_norm ** 2) * 2.0  # factor 2 for v_max

    v = direction.repeat(1, n, 1) * speed
    return torch.cat([v, speed], dim=-1)  # (b, n, 4): vx vy vz | speed


# -------- registry --------
_REGISTRY: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "craft": _direction_craft,
    "nasa": _direction_nasa,
    "crash": _direction_crash,
    "hull": _direction_hull,
    "drivAerML": _direction_drivAerML,
    "hemo": _direction_hemo,
}

_ALIASES: Dict[str, str] = {
    "Craft": "craft",
    "NASA": "nasa",
    "Hull": "hull",
    "Car": "drivAerML",
    "drivAerml": "drivAerML",
    "Hemo": "hemo",
    "HemoPT": "hemo",
}


def get_direction(dynamics_config: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    dynamics_config -> direction(x, cond)
    """
    if dynamics_config in _ALIASES:
        dynamics_config = _ALIASES[dynamics_config]

    if dynamics_config not in _REGISTRY:
        raise ValueError(f"Unknown dynamics_config='{dynamics_config}'. Supported: {list(_REGISTRY.keys())}")

    return _REGISTRY[dynamics_config]
