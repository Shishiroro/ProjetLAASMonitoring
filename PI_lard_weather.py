"""
PI_lard_weather.py  —  XPPython3 plugin for LARD weather injection
===================================================================
Uses the official XPLMWeather API (X-Plane 12.0+) to inject weather.
Communicates with the LARD pipeline via JSON files.

Installation:
  Copy to X-Plane 12/Resources/plugins/XPPython3/PythonPlugins/

Protocol:
  Python pipeline writes weather_command.json  {seq, action, weather}
  Plugin reads, applies via setWeatherAtLocation, writes weather_status.json
"""

import os
import json

try:
    import xp
except ImportError:
    raise RuntimeError("PI_lard_weather requires XPPython3")


# ---------------------------------------------------------------------------
# Exchange directory (same as FlyWithLua bridge for convenience)
# ---------------------------------------------------------------------------
EXCHANGE_DIR = None  # set in XPluginStart
CMD_FILE = None
STS_FILE = None
STS_TMP = None


class PythonInterface:

    def __init__(self):
        self.Name = "LARD Weather"
        self.Sig = "lard.weather.xppython3"
        self.Desc = "LARD weather injection via XPLMWeather API"
        self.flight_loop_id = None
        self.last_ack_seq = -1
        # Cached dataref handles
        self.dr_lat = None
        self.dr_lon = None
        self.dr_elev = None

    # ------------------------------------------------------------------
    # Plugin lifecycle
    # ------------------------------------------------------------------

    def XPluginStart(self):
        global EXCHANGE_DIR, CMD_FILE, STS_FILE, STS_TMP
        try:
            EXCHANGE_DIR = os.path.join(
                xp.getSystemPath(),
                "Resources", "plugins", "FlyWithLua", "Scripts", "lard_exchange"
            )
            CMD_FILE = os.path.join(EXCHANGE_DIR, "weather_command.json")
            STS_FILE = os.path.join(EXCHANGE_DIR, "weather_status.json")
            STS_TMP = os.path.join(EXCHANGE_DIR, "weather_status.tmp")
            os.makedirs(EXCHANGE_DIR, exist_ok=True)
            for f in (CMD_FILE, STS_FILE, STS_TMP):
                try:
                    os.remove(f)
                except FileNotFoundError:
                    pass
        except Exception as e:
            xp.log(f"LARD Weather start error: {e}")
        return self.Name, self.Sig, self.Desc

    def XPluginEnable(self):
        try:
            # Cache dataref handles once
            self.dr_lat = xp.findDataRef("sim/flightmodel/position/latitude")
            self.dr_lon = xp.findDataRef("sim/flightmodel/position/longitude")
            self.dr_elev = xp.findDataRef("sim/flightmodel/position/elevation")

            # Cache datarefs for weather control
            self.dr_use_real_wx = xp.findDataRef("sim/weather/use_real_weather_bool")
            self.dr_rain_pct = xp.findDataRef("sim/weather/rain_percent")
            self.dr_precip = xp.findDataRef("sim/weather/region/rain_percent")

            # Flight loop — phase=0 (before flight model, required for weather API)
            self.flight_loop_id = xp.createFlightLoop(self._tick, phase=0)
            xp.scheduleFlightLoop(self.flight_loop_id, interval=-5)
            xp.log(f"LARD Weather: enabled — exchange dir: {EXCHANGE_DIR}")
        except Exception as e:
            xp.log(f"LARD Weather enable error: {e}")
        return 1

    def XPluginDisable(self):
        try:
            if self.flight_loop_id:
                xp.destroyFlightLoop(self.flight_loop_id)
                self.flight_loop_id = None
        except Exception as e:
            xp.log(f"LARD Weather disable error: {e}")
        xp.log("LARD Weather: disabled")

    def XPluginStop(self):
        pass

    def XPluginReceiveMessage(self, inFrom, inMsg, inParam):
        pass

    # ------------------------------------------------------------------
    # Flight loop callback
    # ------------------------------------------------------------------

    def _tick(self, sinceLast, sinceFlightLoop, counter, refCon):
        try:
            self._process_command()
        except Exception as e:
            xp.log(f"LARD Weather tick error: {e}")
        return -5  # ~5 Hz

    def _process_command(self):
        if CMD_FILE is None:
            return

        # Read command
        try:
            with open(CMD_FILE, "r") as f:
                cmd = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return

        seq = cmd.get("seq")
        if seq is None or seq == self.last_ack_seq:
            return

        action = cmd.get("action", "noop")
        status = {"ack_seq": seq, "ok": True}

        try:
            if action == "set_weather":
                self._apply_weather(cmd.get("weather", {}))
            elif action == "clear_weather":
                self._clear_weather()
            elif action == "noop":
                pass
            else:
                status["ok"] = False
                status["error"] = f"unknown action: {action}"
        except Exception as e:
            status["ok"] = False
            status["error"] = str(e)
            xp.log(f"LARD Weather action error: {e}")

        # Write status atomically
        self._write_status(status)
        self.last_ack_seq = seq

    # ------------------------------------------------------------------
    # Weather injection via XPLMWeather API
    # ------------------------------------------------------------------

    def _get_aircraft_pos(self):
        """Return (lat, lon, elev_msl) of the aircraft."""
        lat = xp.getDatad(self.dr_lat)
        lon = xp.getDatad(self.dr_lon)
        elev = xp.getDataf(self.dr_elev)
        return lat, lon, elev

    def _apply_weather(self, weather):
        """Inject weather at the aircraft's current position."""
        # Disable real weather (METAR) so our injection takes effect
        xp.setDatai(self.dr_use_real_wx, 0)

        lat, lon, elev = self._get_aircraft_pos()

        # Get current weather as a base, then modify it
        # This avoids constructing WeatherInfo_t from scratch
        info = xp.getWeatherAtLocation(lat, lon, elev)

        # Cloud layer 0: our injected clouds
        cloud_base = float(weather.get("cloud_base_msl", 100.0))
        cloud_top = float(weather.get("cloud_top_msl", 600.0))
        cloud_type = float(weather.get("cloud_type", 4.0))  # 4 = overcast
        cloud_coverage = float(weather.get("cloud_coverage", 1.0))

        info.cloud_layers[0].cloud_type = cloud_type
        info.cloud_layers[0].coverage = cloud_coverage
        info.cloud_layers[0].alt_base = cloud_base
        info.cloud_layers[0].alt_top = cloud_top

        # Clear other cloud layers (no parasites)
        for i in range(1, len(info.cloud_layers)):
            info.cloud_layers[i].cloud_type = 0.0
            info.cloud_layers[i].coverage = 0.0

        # Precipitation (both base rate and at-altitude rate)
        precip = float(weather.get("precip_rate", 0.0))
        info.precip_rate = precip
        info.precip_rate_alt = precip

        # Visibility
        info.visibility = float(weather.get("visibility_m", 10000.0))

        # Radius and ceiling
        info.radius_nm = float(weather.get("radius_nm", 50.0))
        info.max_altitude_msl_ft = float(weather.get("max_alt_ft", 50000.0))

        # Apply via context manager (handles begin/end automatically)
        with xp.weatherUpdateContext(updateImmediately=1):
            xp.setWeatherAtLocation(lat, lon, elev, info)

        # Also force rain via datarefs (API precip_rate alone may not trigger visual rain)
        rain_pct = float(weather.get("precip_rate", 0.0)) * 100.0
        try:
            xp.setDataf(self.dr_rain_pct, rain_pct)
        except Exception:
            pass
        try:
            xp.setDataf(self.dr_precip, rain_pct)
        except Exception:
            pass

        xp.log(f"LARD Weather: SET clouds={cloud_base:.0f}-{cloud_top:.0f}m "
               f"type={cloud_type:.0f} cov={cloud_coverage:.1f} "
               f"precip={info.precip_rate:.2f} rain_pct={rain_pct:.0f} "
               f"vis={info.visibility:.0f}m at ({lat:.4f}, {lon:.4f})")

    def _clear_weather(self):
        """Reset to clear skies."""
        # Re-enable real weather
        xp.setDatai(self.dr_use_real_wx, 1)

        lat, lon, elev = self._get_aircraft_pos()

        # Get current, then clear it
        info = xp.getWeatherAtLocation(lat, lon, elev)

        # Clear all clouds
        for i in range(len(info.cloud_layers)):
            info.cloud_layers[i].cloud_type = 0.0
            info.cloud_layers[i].coverage = 0.0

        info.precip_rate = 0.0
        info.visibility = 50000.0
        info.radius_nm = 50.0
        info.max_altitude_msl_ft = 50000.0

        with xp.weatherUpdateContext(updateImmediately=1):
            xp.eraseWeatherAtLocation(lat, lon)
            xp.setWeatherAtLocation(lat, lon, elev, info)

        xp.log("LARD Weather: CLEARED")

    # ------------------------------------------------------------------
    # Status file I/O
    # ------------------------------------------------------------------

    def _write_status(self, status):
        try:
            with open(STS_TMP, "w") as f:
                json.dump(status, f)
            if os.path.exists(STS_FILE):
                os.remove(STS_FILE)
            os.rename(STS_TMP, STS_FILE)
        except Exception as e:
            xp.log(f"LARD Weather write error: {e}")
