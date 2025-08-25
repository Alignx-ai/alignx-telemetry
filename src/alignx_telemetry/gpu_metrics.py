"""
GPU metrics collection for AI workloads using NVIDIA Management Library (NVML).
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any
from opentelemetry import metrics

# Import the new MetricsConfig for consistent configuration
from .metrics import MetricsConfig

logger = logging.getLogger(__name__)

try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logger.warning("pynvml not available. GPU metrics will not be collected.")


class GPUMetrics:
    """
    GPU metrics collector for NVIDIA GPUs with AlignX-compatible metrics.
    """

    def __init__(
        self,
        meter_name: str = "alignx.gpu.metrics",
        config: Optional[MetricsConfig] = None,
    ):
        """
        Initialize GPU metrics collector.

        Args:
            meter_name: Name of the meter for OpenTelemetry metrics
            config: Metrics configuration (defaults to MetricsConfig())
        """
        self.meter = metrics.get_meter(meter_name)
        self.config = config or MetricsConfig()
        self._nvml_initialized = False
        self._gpu_count = 0

        # Only initialize metrics if GPU monitoring is enabled
        if self.config.enable_gpu_monitoring:
            self._initialize_metrics()
            self._initialize_nvml()

    def _initialize_metrics(self) -> None:
        """Initialize GPU metrics instruments with AlignX naming."""

        # GPU Utilization Metrics
        self.gpu_utilization = self.meter.create_histogram(
            "gpu_utilization",
            description="GPU utilization percentage (0-100)",
            unit="percent",
        )

        self.gpu_memory_utilization = self.meter.create_histogram(
            "gpu_memory_utilization",
            description="GPU memory utilization percentage (0-100)",
            unit="percent",
        )

        # GPU Memory Metrics
        self.gpu_memory_used = self.meter.create_histogram(
            "gpu_memory_used",
            description="GPU memory used in bytes",
            unit="bytes",
        )

        self.gpu_memory_free = self.meter.create_histogram(
            "gpu_memory_free",
            description="GPU memory free in bytes",
            unit="bytes",
        )

        self.gpu_memory_total = self.meter.create_histogram(
            "gpu_memory_total",
            description="GPU memory total in bytes",
            unit="bytes",
        )

        # GPU Temperature and Power Metrics
        self.gpu_temperature = self.meter.create_histogram(
            "gpu_temperature",
            description="GPU temperature in Celsius",
            unit="celsius",
        )

        self.gpu_power_draw = self.meter.create_histogram(
            "gpu_power_draw",
            description="GPU power draw in watts",
            unit="watt",
        )

        self.gpu_power_limit = self.meter.create_histogram(
            "gpu_power_limit",
            description="GPU power limit in watts",
            unit="watt",
        )

        self.gpu_fan_speed = self.meter.create_histogram(
            "gpu_fan_speed",
            description="GPU fan speed percentage (0-100)",
            unit="percent",
        )

        # Additional GPU Metrics
        self.gpu_enc_utilization = self.meter.create_histogram(
            "gpu_enc_utilization",
            description="GPU encoder utilization percentage (0-100)",
            unit="percent",
        )

        self.gpu_dec_utilization = self.meter.create_histogram(
            "gpu_dec_utilization",
            description="GPU decoder utilization percentage (0-100)",
            unit="percent",
        )

    def _initialize_nvml(self) -> None:
        """Initialize NVML for GPU monitoring."""
        if not NVML_AVAILABLE:
            logger.warning("NVML not available, GPU metrics disabled")
            return

        try:
            pynvml.nvmlInit()
            self._gpu_count = pynvml.nvmlDeviceGetCount()
            self._nvml_initialized = True
            logger.info(
                f"NVML initialized successfully. Found {self._gpu_count} GPU(s)"
            )
        except Exception as e:
            logger.error(f"Failed to initialize NVML: {e}")
            self._nvml_initialized = False

    def _get_gpu_attributes(self, gpu_index: int) -> Dict[str, Any]:
        """
        Get GPU attributes for a specific GPU with AlignX standardization.

        Args:
            gpu_index: GPU index

        Returns:
            Dictionary of GPU attributes
        """
        if not self._nvml_initialized:
            return {}

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

            # Get GPU name
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8")

            # Get GPU UUID
            gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(gpu_uuid, bytes):
                gpu_uuid = gpu_uuid.decode("utf-8")

            return {
                "gpu_index": str(gpu_index),
                "gpu_name": gpu_name,
                "gpu_uuid": gpu_uuid,
                "telemetry_sdk_name": "alignx",
                "telemetry_sdk_version": "1.0.0",
                "gen_ai_application_name": self.config.service_name,
                "gen_ai_environment": self.config.environment,
            }
        except Exception as e:
            logger.error(f"Error getting GPU attributes for GPU {gpu_index}: {e}")
            return {}

    def collect_gpu_metrics(self) -> None:
        """Collect metrics for all available GPUs."""
        if not self.config.enable_gpu_monitoring or not self._nvml_initialized:
            return

        for gpu_index in range(self._gpu_count):
            self._collect_single_gpu_metrics(gpu_index)

    def _collect_single_gpu_metrics(self, gpu_index: int) -> None:
        """
        Collect metrics for a single GPU.

        Args:
            gpu_index: Index of the GPU to collect metrics for
        """
        if not self._nvml_initialized:
            return

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            attributes = self._get_gpu_attributes(gpu_index)

            if not attributes:
                return

            # GPU Utilization
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_utilization.record(utilization.gpu, attributes)
                self.gpu_memory_utilization.record(utilization.memory, attributes)
            except Exception as e:
                logger.debug(f"Could not get utilization for GPU {gpu_index}: {e}")

            # Memory Information
            try:
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.gpu_memory_used.record(memory_info.used, attributes)
                self.gpu_memory_free.record(memory_info.free, attributes)
                self.gpu_memory_total.record(memory_info.total, attributes)
            except Exception as e:
                logger.debug(f"Could not get memory info for GPU {gpu_index}: {e}")

            # Temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                self.gpu_temperature.record(temperature, attributes)
            except Exception as e:
                logger.debug(f"Could not get temperature for GPU {gpu_index}: {e}")

            # Power Information
            try:
                power_draw = (
                    pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                )  # Convert to watts
                self.gpu_power_draw.record(power_draw, attributes)
            except Exception as e:
                logger.debug(f"Could not get power usage for GPU {gpu_index}: {e}")

            try:
                power_limit = (
                    pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1]
                    / 1000.0
                )
                self.gpu_power_limit.record(power_limit, attributes)
            except Exception as e:
                logger.debug(f"Could not get power limit for GPU {gpu_index}: {e}")

            # Fan Speed
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                self.gpu_fan_speed.record(fan_speed, attributes)
            except Exception as e:
                logger.debug(f"Could not get fan speed for GPU {gpu_index}: {e}")

            # Encoder/Decoder Utilization
            try:
                encoder_util = pynvml.nvmlDeviceGetEncoderUtilization(handle)[0]
                self.gpu_enc_utilization.record(encoder_util, attributes)
            except Exception as e:
                logger.debug(
                    f"Could not get encoder utilization for GPU {gpu_index}: {e}"
                )

            try:
                decoder_util = pynvml.nvmlDeviceGetDecoderUtilization(handle)[0]
                self.gpu_dec_utilization.record(decoder_util, attributes)
            except Exception as e:
                logger.debug(
                    f"Could not get decoder utilization for GPU {gpu_index}: {e}"
                )

        except Exception as e:
            logger.error(f"Error collecting metrics for GPU {gpu_index}: {e}")

    def shutdown(self) -> None:
        """Shutdown NVML."""
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
                logger.info("NVML shutdown successfully")
            except Exception as e:
                logger.error(f"Error during NVML shutdown: {e}")

    def is_available(self) -> bool:
        """Check if GPU monitoring is available and enabled."""
        return (
            NVML_AVAILABLE
            and self.config.enable_gpu_monitoring
            and self._nvml_initialized
            and self._gpu_count > 0
        )

    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all available GPUs.

        Returns:
            List of dictionaries containing GPU information
        """
        if not self.is_available():
            return []

        gpu_info = []
        for gpu_index in range(self._gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode("utf-8")

                gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(gpu_uuid, bytes):
                    gpu_uuid = gpu_uuid.decode("utf-8")

                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                gpu_info.append(
                    {
                        "index": gpu_index,
                        "name": gpu_name,
                        "uuid": gpu_uuid,
                        "memory_total": memory_info.total,
                        "memory_used": memory_info.used,
                        "memory_free": memory_info.free,
                    }
                )
            except Exception as e:
                logger.error(f"Error getting info for GPU {gpu_index}: {e}")

        return gpu_info


# Global GPU metrics collector
_gpu_metrics_collector = None
_collection_thread = None
_collection_stop_event = None


def start_gpu_metrics_collection(
    collection_interval: float = 10.0, config: Optional[MetricsConfig] = None
) -> bool:
    """
    Start GPU metrics collection in a background thread.

    Args:
        collection_interval: How often to collect metrics (in seconds)
        config: Metrics configuration (defaults to MetricsConfig())

    Returns:
        True if collection started successfully, False otherwise
    """
    global _gpu_metrics_collector, _collection_thread, _collection_stop_event

    config = config or MetricsConfig()

    # Check if GPU monitoring is disabled
    if not config.enable_gpu_monitoring:
        logger.info("GPU monitoring is disabled via configuration")
        return False

    # Check if already running
    if _collection_thread and _collection_thread.is_alive():
        logger.warning("GPU metrics collection is already running")
        return True

    # Initialize GPU metrics collector
    try:
        _gpu_metrics_collector = GPUMetrics(config=config)
        if not _gpu_metrics_collector.is_available():
            logger.warning("GPU monitoring not available")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize GPU metrics collector: {e}")
        return False

    # Start collection thread
    _collection_stop_event = threading.Event()

    def collection_loop():
        """Main collection loop."""
        logger.info(
            f"Started GPU metrics collection (interval: {collection_interval}s)"
        )
        while not _collection_stop_event.wait(collection_interval):
            try:
                _gpu_metrics_collector.collect_gpu_metrics()
            except Exception as e:
                logger.error(f"Error during GPU metrics collection: {e}")

        logger.info("GPU metrics collection stopped")

    _collection_thread = threading.Thread(target=collection_loop, daemon=True)
    _collection_thread.start()

    return True


def stop_gpu_metrics_collection() -> None:
    """Stop GPU metrics collection."""
    global _gpu_metrics_collector, _collection_thread, _collection_stop_event

    if _collection_stop_event:
        _collection_stop_event.set()

    if _collection_thread and _collection_thread.is_alive():
        _collection_thread.join(timeout=5.0)

    if _gpu_metrics_collector:
        _gpu_metrics_collector.shutdown()
        _gpu_metrics_collector = None

    _collection_thread = None
    _collection_stop_event = None


def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Get information about available GPUs.

    Returns:
        List of GPU information dictionaries
    """
    if _gpu_metrics_collector:
        return _gpu_metrics_collector.get_gpu_info()

    # Create temporary collector for info gathering
    try:
        temp_collector = GPUMetrics()
        if temp_collector.is_available():
            info = temp_collector.get_gpu_info()
            temp_collector.shutdown()
            return info
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")

    return []


def is_gpu_available() -> bool:
    """
    Check if GPU monitoring is available.

    Returns:
        True if GPU monitoring is available, False otherwise
    """
    if _gpu_metrics_collector:
        return _gpu_metrics_collector.is_available()

    # Create temporary collector to check availability
    try:
        temp_collector = GPUMetrics()
        available = temp_collector.is_available()
        temp_collector.shutdown()
        return available
    except Exception:
        return False
