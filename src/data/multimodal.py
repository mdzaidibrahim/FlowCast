"""Multi-Modal Domain Extension for FlowCast.

Calculates derived environmental and urban infrastructure indicators:
1. Dynamic Carbon Footprint Estimation (kg/metric tons of CO2 based on congestion non-linearities)
2. EV Charging Station & Smart Grid Demand Surge Index (kW / % Surge)
3. Prescriptive Green-Wave Speed Recommendation (km/h advisory to clear bottleneck shockwaves)
"""

from __future__ import annotations

from typing import Dict, Tuple, Union
import numpy as np


# Carbon Emission Factors (EPA / EEA arterial corridor models)
# Idle / heavy stop-and-go emission factor is ~2.8x free-flow emissions per minute
BASE_CO2_GRAMS_PER_KM = 165.0  # Average passenger vehicle CO2 g/km at optimal 60 km/h
HIGHWAY_SEGMENT_LENGTH_KM = 2.5  # Typical detector spacing / catchment area in km
MAX_CAPACITY_VPH = 2200.0  # Max vehicles per hour per lane standard capacity


def compute_carbon_emissions(
    volumes: np.ndarray,
    num_lanes: Union[np.ndarray, int] = 3,
    time_horizon_minutes: float = 15.0,
) -> np.ndarray:
    """Calculate estimated CO2 emissions in kilograms for the given traffic volume.

    Uses a non-linear congestion penalty function:
    Under high normalized volume (> 0.7), average vehicle speed drops dramatically,
    leading to idling and heightened fuel consumption / carbon release.

    Args:
        volumes: Normalized traffic volume array [0.0, 1.0] of any shape.
        num_lanes: Number of lanes per sensor station (default: 3).
        time_horizon_minutes: Duration of forecasting window (default: 15.0 min).

    Returns:
        Estimated CO2 emissions in kg emitted across the corridor section.
    """
    # Estimated vehicle count passing through in 15 mins:
    # volume (0..1) * capacity_per_lane * num_lanes * (15/60)
    vehicles = volumes * MAX_CAPACITY_VPH * num_lanes * (time_horizon_minutes / 60.0)

    # Congestion multiplier: Exponential increase as corridor saturates towards capacity
    # Speed proxy: free_flow = 100 km/h (vol < 0.3), drops to 25 km/h at vol > 0.85
    speed_factor = 1.0 + 1.8 * np.power(np.clip(volumes, 0.0, 1.0), 2.5)

    # Total CO2 in kg: vehicles * distance (km) * (grams_per_km * speed_factor) / 1000
    co2_kg = (vehicles * HIGHWAY_SEGMENT_LENGTH_KM * (BASE_CO2_GRAMS_PER_KM * speed_factor)) / 1000.0
    return np.round(co2_kg, 2)


def compute_ev_grid_surge_index(
    volumes: np.ndarray,
    ctx_hour: Union[np.ndarray, int] = 17,
) -> np.ndarray:
    """Calculate the EV Fast-Charging Station & Smart Grid Surge Pressure Index (0-100%).

    Fast charging hub arrival rates heavily correlate with corridor traffic volume,
    amplified during late afternoon/evening commute hours (16:00 - 20:00).

    Args:
        volumes: Normalized traffic volume array [0.0, 1.0].
        ctx_hour: Hour of the day (0 to 23).

    Returns:
        Surge Pressure Index scaled between 0.0% and 100.0%.
    """
    # Peak hour weight multiplier
    if isinstance(ctx_hour, np.ndarray):
        is_peak = ((ctx_hour >= 7) & (ctx_hour <= 10)) | ((ctx_hour >= 16) & (ctx_hour <= 20))
        hour_multiplier = np.where(is_peak, 1.35, 0.85)
    else:
        is_peak = (7 <= ctx_hour <= 10) or (16 <= ctx_hour <= 20)
        hour_multiplier = 1.35 if is_peak else 0.85

    # EV penetration assumption (approx 12-18% of fleet looking for corridor chargers)
    surge_raw = (volumes * 100.0) * hour_multiplier
    surge_index = np.clip(surge_raw, 0.0, 100.0)
    return np.round(surge_index, 1)


def compute_green_wave_advisory(
    volumes: np.ndarray,
) -> Dict[str, Union[np.ndarray, str]]:
    """Determine dynamic speed harmonization and green-wave signal recommendations.

    To eliminate upstream shockwave propagation when downstream bottleneck approaches:
    - Low volume (<= 0.35): Normal posted limit (100 km/h) - Free Flow
    - Moderate volume (0.35 < vol <= 0.70): Speed advisory (70-80 km/h) - Variable Speed Limit
    - High volume (> 0.70): Congestion metering (45-55 km/h) - Shockwave Suppression & Ramp Metering

    Args:
        volumes: Normalized volume array [0.0, 1.0] across sensors.

    Returns:
        Dict with recommended advisory speeds (km/h) and operational actions.
    """
    recommended_speed = np.zeros_like(volumes, dtype=np.float32)

    # Vectorized thresholds
    mask_low = volumes <= 0.35
    mask_mod = (volumes > 0.35) & (volumes <= 0.70)
    mask_high = volumes > 0.70

    recommended_speed[mask_low] = 100.0
    recommended_speed[mask_mod] = 75.0 - (volumes[mask_mod] - 0.35) * 25.0  # Linear ramp down to 65
    recommended_speed[mask_high] = 50.0 - (volumes[mask_high] - 0.70) * 20.0  # Drop to 40-50

    return {
        "recommended_speed_kmh": np.round(np.clip(recommended_speed, 40.0, 100.0), 1),
        "metering_active": mask_high,
    }
