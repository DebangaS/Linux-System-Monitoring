"""
GPU and Graphics Hardware Monitoring
Author: Member 2 (fixed & completed)
Provides:
 - GPUMonitor: GPU statistics (via NVIDIA pynvml when available; graceful fallback otherwise)
 - HardwareMonitor: temperature, fan, battery info via psutil (when available)
"""
from datetime import datetime
from typing import Dict, List, Optional
import logging
import psutil

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GPUMonitor:
    """GPU monitoring and statistics collection"""

    def __init__(self):
        self.pynvml = None
        self.nvidia_available = self._check_nvidia_ml()
        self.gpu_info_cache: Dict = {}
        self.last_update: Optional[datetime] = None

    def _check_nvidia_ml(self) -> bool:
        """Check if NVIDIA NVML (pynvml) library is available and initialize it."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            return True
        except (ImportError, Exception) as e:
            logger.info(f"NVIDIA NVML not available or failed to init: {e}")
            self.pynvml = None
            return False

    def get_gpu_info(self) -> Dict:
        """Get detailed GPU information. Uses pynvml if available; otherwise fallback."""
        if not self.nvidia_available or self.pynvml is None:
            return self._get_fallback_gpu_info()

        pynvml = self.pynvml
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            gpus = []
            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                except Exception as e:
                    logger.warning(f"Could not get handle for GPU {i}: {e}")
                    continue

                # Basic info
                try:
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", errors="replace")
                except Exception:
                    name = "Unknown"

                # Memory info
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_total = getattr(mem_info, "total", None)
                    mem_used = getattr(mem_info, "used", None)
                    mem_free = getattr(mem_info, "free", None)
                except Exception:
                    mem_total = mem_used = mem_free = None

                # Temperature
                temperature = None
                try:
                    temp_const = getattr(pynvml, "NVML_TEMPERATURE_GPU", None)
                    if temp_const is not None:
                        temperature = pynvml.nvmlDeviceGetTemperature(handle, temp_const)
                    else:
                        # try default constant name (best-effort)
                        temperature = pynvml.nvmlDeviceGetTemperature(handle, 0)
                except Exception:
                    temperature = None

                # Utilization
                gpu_util = None
                memory_util = None
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = getattr(util, "gpu", None)
                    memory_util = getattr(util, "memory", None)
                except Exception:
                    gpu_util = memory_util = None

                # Power draw (in watts)
                power = None
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    if power_mw is not None:
                        power = float(power_mw) / 1000.0
                except Exception:
                    power = None

                # Clock speeds
                graphics_clock = None
                memory_clock = None
                try:
                    clk_graphics_const = getattr(pynvml, "NVML_CLOCK_GRAPHICS", None)
                    clk_mem_const = getattr(pynvml, "NVML_CLOCK_MEM", None) or getattr(pynvml, "NVML_CLOCK_MEMORIES", None)
                    if clk_graphics_const is not None:
                        graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, clk_graphics_const)
                    if clk_mem_const is not None:
                        memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, clk_mem_const)
                except Exception:
                    graphics_clock = memory_clock = None

                # Fan speed (percentage)
                fan_speed = None
                try:
                    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                except Exception:
                    fan_speed = None

                gpu_data = {
                    "index": i,
                    "name": name,
                    "memory": {
                        "total": mem_total,
                        "used": mem_used,
                        "free": mem_free,
                        "utilization_percent": memory_util,
                    },
                    "utilization": {
                        "gpu_percent": gpu_util,
                        "memory_percent": memory_util,
                    },
                    "temperature_celsius": temperature,
                    "power_draw_watts": power,
                    "clocks": {
                        "graphics_mhz": graphics_clock,
                        "memory_mhz": memory_clock,
                    },
                    "fan_speed_percent": fan_speed,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                gpus.append(gpu_data)

            # System-level info
            driver_version = None
            cuda_version = None
            try:
                dv = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(dv, bytes):
                    driver_version = dv.decode("utf-8", errors="replace")
                else:
                    driver_version = str(dv)
            except Exception:
                driver_version = None

            try:
                # NVML sometimes exposes a CUDA driver-related function; try best-effort
                cv = getattr(pynvml, "nvmlSystemGetCudaDriverVersion", None)
                if cv:
                    cuda_version = cv()
                    if isinstance(cuda_version, bytes):
                        cuda_version = cuda_version.decode("utf-8", errors="replace")
            except Exception:
                cuda_version = None

            result = {
                "gpus": gpus,
                "total_gpus": device_count,
                "driver_version": driver_version or "unknown",
                "cuda_version": cuda_version or "unknown",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            # Cache metadata
            self.gpu_info_cache = result
            self.last_update = datetime.utcnow()
            return result

        except Exception as e:
            logger.error(f"GPU monitoring error: {e}", exc_info=True)
            return self._get_fallback_gpu_info()

    def _get_fallback_gpu_info(self) -> Dict:
        """Fallback GPU info when NVIDIA NVML is not available"""
        return {
            "gpus": [],
            "total_gpus": 0,
            "driver_version": "unknown",
            "cuda_version": "unknown",
            "error": "GPU monitoring not available",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def get_gpu_processes(self) -> List[Dict]:
        """Get processes using GPU (best-effort). Returns empty list if NVML not available."""
        if not self.nvidia_available or self.pynvml is None:
            return []

        pynvml = self.pynvml
        processes = []
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                except Exception:
                    logger.warning(f"Could not get handle for GPU {i}")
                    continue

                # Try both compute and graphics process listing (API names differ across NVML versions)
                procs = []
                try:
                    # newer NVML: nvmlDeviceGetComputeRunningProcesses or _v2 variant
                    getter = getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None) or getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v2", None)
                    if getter is not None:
                        procs = getter(handle)
                except Exception:
                    procs = []

                # If no procs found, try graphics running processes (best-effort)
                if not procs:
                    try:
                        getter = getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None) or getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses_v2", None)
                        if getter is not None:
                            procs = getter(handle)
                    except Exception:
                        procs = []

                for proc in procs:
                    try:
                        pid = getattr(proc, "pid", None) or getattr(proc, "pid", None)
                        usedGpuMemory = getattr(proc, "usedGpuMemory", None)
                        if pid is None:
                            continue
                        # psutil process info (may raise if process has ended or permission denied)
                        try:
                            ps_proc = psutil.Process(pid)
                            process_data = {
                                "gpu_index": i,
                                "pid": pid,
                                "name": ps_proc.name(),
                                "username": ps_proc.username(),
                                "gpu_memory_used": usedGpuMemory,
                                "cpu_percent": ps_proc.cpu_percent(interval=0.0),
                                "memory_percent": ps_proc.memory_percent(),
                                "create_time": ps_proc.create_time(),
                                "status": ps_proc.status(),
                            }
                            processes.append(process_data)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    except Exception as e:
                        logger.warning(f"Error processing GPU process entry on device {i}: {e}")
                        continue

            return processes

        except Exception as e:
            logger.error(f"GPU process monitoring error: {e}", exc_info=True)
            return []


class HardwareMonitor:
    """Hardware sensors and temperature monitoring using psutil where available."""

    def __init__(self):
        self.sensors_available = self._check_sensors()

    def _check_sensors(self) -> bool:
        """Check if psutil exposes sensor functions on this platform."""
        try:
            # psutil may raise AttributeError if sensors_* are not implemented
            _ = psutil.sensors_temperatures  # type: ignore[attr-defined]
            return True
        except Exception:
            return False

    def get_temperature_sensors(self) -> Dict:
        """Get temperature sensor readings (best-effort)."""
        if not self.sensors_available:
            return {"sensors": [], "error": "Temperature sensors not available"}

        try:
            temps = psutil.sensors_temperatures()  # type: ignore[attr-defined]
            sensor_data = {}
            for sensor_name, sensor_list in temps.items():
                sensor_readings = []
                for sensor in sensor_list:
                    reading = {
                        "label": getattr(sensor, "label", None) or "Unknown",
                        "current_celsius": getattr(sensor, "current", None),
                        "high_celsius": getattr(sensor, "high", None),
                        "critical_celsius": getattr(sensor, "critical", None),
                    }
                    sensor_readings.append(reading)
                sensor_data[sensor_name] = sensor_readings
            return {"sensors": sensor_data, "timestamp": datetime.utcnow().isoformat() + "Z"}
        except Exception as e:
            logger.error(f"Temperature sensor error: {e}", exc_info=True)
            return {"sensors": [], "error": str(e)}

    def get_fan_sensors(self) -> Dict:
        """Get fan speed sensor readings (best-effort)."""
        try:
            fans = psutil.sensors_fans()  # type: ignore[attr-defined]
            fan_data = {}
            for fan_name, fan_list in (fans or {}).items():
                fan_readings = []
                for fan in fan_list:
                    reading = {
                        "label": getattr(fan, "label", None) or "Unknown",
                        "current_rpm": getattr(fan, "current", None),
                    }
                    fan_readings.append(reading)
                fan_data[fan_name] = fan_readings
            return {"fans": fan_data, "timestamp": datetime.utcnow().isoformat() + "Z"}
        except Exception as e:
            return {"fans": [], "error": "Fan sensors not available: " + str(e)}

    def get_battery_info(self) -> Dict:
        """Get battery information for laptops (if available)."""
        try:
            battery = psutil.sensors_battery()  # type: ignore[attr-defined]
            if battery is None:
                return {"battery": None, "error": "No battery detected"}
            secsleft = battery.secsleft
            time_remaining_hours = None
            if secsleft not in (psutil.POWER_TIME_UNKNOWN, psutil.POWER_TIME_UNLIMITED) and isinstance(secsleft, (int, float)):
                try:
                    time_remaining_hours = float(secsleft) / 3600.0
                except Exception:
                    time_remaining_hours = None
            return {
                "battery": {
                    "percent": battery.percent,
                    "secsleft": secsleft,
                    "power_plugged": battery.power_plugged,
                    "time_remaining_hours": time_remaining_hours,
                },
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        except Exception as e:
            logger.error(f"Battery info error: {e}", exc_info=True)
            return {"battery": None, "error": str(e)}


# Global monitor instances (convenience)
gpu_monitor = GPUMonitor()
hardware_monitor = HardwareMonitor()


if __name__ == "__main__":
    # Quick smoke-test for local run
    import json
    logging.basicConfig(level=logging.INFO)

    print("GPU Info:")
    info = gpu_monitor.get_gpu_info()
    print(json.dumps(info, indent=2))

    print("\nGPU Processes (if any):")
    procs = gpu_monitor.get_gpu_processes()
    print(json.dumps(procs, indent=2))

    print("\nTemperature Sensors:")
    temp = hardware_monitor.get_temperature_sensors()
    print(json.dumps(temp, indent=2))

    print("\nFan Sensors:")
    fans = hardware_monitor.get_fan_sensors()
    print(json.dumps(fans, indent=2))

    print("\nBattery Info:")
    batt = hardware_monitor.get_battery_info()
    print(json.dumps(batt, indent=2))
