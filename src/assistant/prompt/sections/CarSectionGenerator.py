"""The CURRENT CAR INFO section: one line per car of every ready VehicleProvider — Tesla
beside whatever other brand the house has — with the car's vehicle_id, whether it is parked or
driving, where it is (Home, or an address), its route or how far from home it is, the cabin
temperature, whether the climate is on, the charge and the range — rounded and held (Held.py), so a parked or charging
car's line stays the same text for hours and the prompt's cache with it. Refreshed every five minutes while every car is parked and every
minute while one is driving; a parked car's minutes from home are asked for every ten minutes
and held through a one-minute traffic wobble; a brand that fails gets an error on its car's
line and the others stay. Without a ready provider the section is the setup hint."""
import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests

from assistant.integration.VehicleProvider import VehicleProvider, VehicleSpec, VehicleState, active, Fleet, no_provider_hint
from assistant.tools.Geocoding import reverse_geocode
from common.ConfigManager import config
from common.logging import info, trace

from ..Held import HeldValues
from ..PromptSectionGenerator import PromptSectionGenerator


@dataclass(frozen=True)
class _ParkedEta:
    """The ETA to home shown for a parked car, with when and where it was fetched."""
    minutes: int
    fetched_at: float
    latitude: float
    longitude: float


class CarSectionGenerator(PromptSectionGenerator):
    """Refreshes the car section from every ready car integration; runs only while one is on."""
    key = "car_info"
    heading = "# CURRENT CAR INFO"
    intro = ("The family's cars, every brand together. `vehicle_id` is what the `cars` toolkit's commands take. "
             "A parked car's charge is rounded to 5% and its range to 50 km, a charging car's charge to 10%; "
             "a car marked asleep or offline shows the brand's last cached data.")
    needs = VehicleProvider

    HOME_RADIUS_METERS = 500
    UPDATE_INTERVAL_DRIVING = 60
    UPDATE_INTERVAL_PARKED = 300
    UPDATE_INTERVAL_NOT_READY = 30      # a provider is on but not configured yet: look again soon after setup
    # A parked car's ETA to home moves with the traffic alone, and the Distance Matrix answer
    # wobbles by a minute between calls — each wobble a diff of the section. So a parked car
    # is asked for every ten minutes, and the line keeps its value until the answer moves by
    # the dead band. A driving car's ETA is live.
    PARKED_ETA_REFRESH_S = 600
    PARKED_ETA_DEAD_BAND_MIN = 2
    PARKED_ETA_SAME_SPOT_M = 100
    # The brands report the cabin in tenths, the charge in percents and the range to the km —
    # a parked car in the sun or on a charger would rewrite its line every poll. The line shows
    # whole degrees; while the car is parked 5% steps of charge and 50 km steps of range; while
    # it charges 10% steps and no range (a note at plug-in, one at the end and a handful in
    # between instead of one per poll — 2026-10-19). A driving car's charge and range are
    # exact, its line changes anyway.
    CABIN_STEP_C = 1
    PARKED_ENERGY_STEP = 5
    CHARGING_ENERGY_STEP = 10
    PARKED_RANGE_STEP_KM = 50

    def __init__(self, prompt_manager):
        super().__init__(prompt_manager, "CarInfo")
        # Optional: without a home location the cars are placed by address only.
        self.home_lat = config.get_home_latitude()
        self.home_lon = config.get_home_longitude()
        self._any_car_driving = False
        self._ready = False
        self._parked_etas: dict[str, _ParkedEta] = {}
        self._cabin = HeldValues(self.CABIN_STEP_C)
        self._parked_energy = HeldValues(self.PARKED_ENERGY_STEP)
        self._charging_energy = HeldValues(self.CHARGING_ENERGY_STEP)
        self._parked_range = HeldValues(self.PARKED_RANGE_STEP_KM)

    def _start_monitoring(self):
        while not self._stop_event.is_set():
            self._update_content()
            self._wait(self._get_update_interval())

    def _get_update_interval(self) -> int:
        """Driving status is carried over from the last fetch pass — fresh state calls here
        would double the traffic to the brands' clouds for nothing."""
        if not self._ready:
            return self.UPDATE_INTERVAL_NOT_READY
        return self.UPDATE_INTERVAL_DRIVING if self._any_car_driving else self.UPDATE_INTERVAL_PARKED

    def _fetch_content(self) -> str:
        providers = active()
        self._ready = bool(providers)
        if not providers:
            self._any_car_driving = False
            return f"({no_provider_hint()})"

        trace(f"Fetching car information from {', '.join(p.name for p in providers)}...")
        self._any_car_driving = False
        lines = []
        for provider, spec in Fleet(providers).cars():
            try:
                lines.append(self._render(spec, provider.vehicle_state(spec)))
            except Exception as e:
                info(f"CarInfo: {provider.name} failed for {spec.label}: {e}")
                lines.append(f"{spec.label} (vehicle_id: {spec.id}): Error fetching data - {e}")
        if not lines:
            info("CarInfo: the providers list no cars")
            return "No car information available."
        trace(f"{len(lines)} lines of car information fetched.")
        return "\n".join(lines)

    def _render(self, spec: VehicleSpec, state: VehicleState) -> str:
        """One car's line for the prompt."""
        is_driving = state.is_driving
        if is_driving or state.in_gear:
            self._any_car_driving = True
        status = "Driving" if is_driving else "Charging" if state.charging else "Parked"
        speed_text = f"{state.speed_kmh} km/h" if is_driving else ""

        has_position = state.latitude is not None and state.longitude is not None
        if has_position and self.home_lat is not None and self.home_lon is not None:
            distance_m = self._haversine_distance(self.home_lat, self.home_lon, state.latitude, state.longitude)
            if distance_m <= self.HOME_RADIUS_METERS:
                location_text = "Home"
            else:
                location_text = reverse_geocode(state.latitude, state.longitude) or "Not home"
        else:
            location_text = "Location unknown"
        is_at_home = location_text == "Home"

        # A route left loaded in the nav of a parked car would contradict "Parked" (Tesla keeps
        # the route until it is cancelled), so only a car in gear is navigating.
        has_route = state.minutes_to_arrival is not None and state.destination and (is_driving or state.in_gear)
        if has_route:
            eta_text = f"Navigating to {state.destination}, ETA {round(state.minutes_to_arrival)} min"
        elif not is_at_home and has_position:
            eta_minutes = self._eta_to_home(spec, state)
            eta_text = f"~{eta_minutes} min from home" if eta_minutes is not None else ""
        else:
            eta_text = ""

        cabin_c = self._cabin.show(spec.id, state.cabin_temp_c)
        cabin_text = f"Cabin {cabin_c}°C" if cabin_c is not None else ""
        climate_text = "" if state.climate_on is None else "climate on" if state.climate_on else "climate off"
        energy, unit = (state.battery_percent, "battery") if state.battery_percent is not None else (state.fuel_percent, "fuel")
        moving = is_driving or state.in_gear
        if energy is None:
            self._parked_energy.forget(spec.id)
            self._charging_energy.forget(spec.id)
            energy_text = "Battery unknown"
        elif moving:
            self._parked_energy.forget(spec.id)
            self._charging_energy.forget(spec.id)
            energy_text = f"({energy}% {unit})"
        elif state.charging:
            self._parked_energy.forget(spec.id)
            energy_text = f"(~{self._charging_energy.show(spec.id, energy)}% {unit})"
        else:
            self._charging_energy.forget(spec.id)
            energy_text = f"(~{self._parked_energy.show(spec.id, energy)}% {unit})"
        if state.range_km is None or state.charging:
            self._parked_range.forget(spec.id)
            range_text = ""
        elif moving:
            self._parked_range.forget(spec.id)
            range_text = f"{round(state.range_km)} km range"
        else:
            range_text = f"~{self._parked_range.show(spec.id, state.range_km)} km range"

        # The brand serves a cached snapshot while the car sleeps — flag it so the LLM knows
        # the data may be stale.
        asleep_text = ""
        if state.connectivity is not None and state.connectivity != "online":
            if state.data_time:
                # To the hour: the snapshot's minute would rewrite the line each time the car stirs.
                asleep_text = f"[car is {state.connectivity}, data from ~{datetime.fromtimestamp(state.data_time).strftime('%H:00')}]"
            else:
                asleep_text = f"[car is {state.connectivity}, data may be stale]"

        parts = [f"{spec.label} (vehicle_id: {spec.id}):", status, speed_text, location_text, eta_text,
                 cabin_text, climate_text, energy_text, range_text, asleep_text]
        return " ".join(p for p in parts if p)

    def _eta_to_home(self, spec: VehicleSpec, state: VehicleState) -> Optional[int]:
        """The minutes from home for the line: live for a moving car, held for a parked one
        (see PARKED_ETA_REFRESH_S)."""
        if state.is_driving or state.in_gear:
            self._parked_etas.pop(spec.id, None)
            return self._get_eta_to_home(state.latitude, state.longitude)
        held = self._parked_etas.get(spec.id)
        if held and self._haversine_distance(held.latitude, held.longitude, state.latitude, state.longitude) > self.PARKED_ETA_SAME_SPOT_M:
            trace(f"CarInfo: {spec.label} is parked somewhere else, its held ETA is dropped")
            held = None
        now = time.time()
        if held and now - held.fetched_at < self.PARKED_ETA_REFRESH_S:
            return held.minutes
        fetched = self._get_eta_to_home(state.latitude, state.longitude)
        if fetched is None:
            # The held value stands on a failed fetch; the next pass asks again.
            return held.minutes if held else None
        minutes = fetched
        if held and abs(fetched - held.minutes) < self.PARKED_ETA_DEAD_BAND_MIN:
            trace(f"CarInfo: {spec.label} ETA fetched as {fetched} min, kept at {held.minutes} min")
            minutes = held.minutes
        self._parked_etas[spec.id] = _ParkedEta(minutes, now, state.latitude, state.longitude)
        return minutes

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """The great circle distance in meters between two points given in decimal degrees."""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2 * math.asin(math.sqrt(a)) * 6371000

    def _get_eta_to_home(self, latitude: float, longitude: float) -> Optional[int]:
        """ETA to home in minutes from the Google Distance Matrix API with traffic; None on
        error or without an API key."""
        try:
            api_key = config.get('api_keys.google_maps', "")
            if not api_key or self.home_lat is None or self.home_lon is None:
                return None

            url = "https://maps.googleapis.com/maps/api/distancematrix/json"
            params = {
                "origins": f"{latitude},{longitude}",
                "destinations": f"{self.home_lat},{self.home_lon}",
                "mode": "driving",
                "departure_time": "now",
                "key": api_key,
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "OK":
                    elements = data.get("rows", [{}])[0].get("elements", [{}])[0]
                    if elements.get("status") == "OK":
                        duration = elements.get("duration_in_traffic") or elements.get("duration")
                        if duration:
                            trace("Successfully fetched ETA to home.")
                            return round(duration.get("value", 0) / 60)
            return None
        except Exception:
            return None
