"""Generic metrics abstraction for the RAW library.

Defines a lightweight :class:`Metrics` protocol that any subsystem (storage,
LLMs, agents, …) can use to emit counters, histograms, and gauges — without
coupling the library to a specific metrics backend.

**Usage (library internals):**

.. code-block:: python

    from RAW.utils.metrics import Metrics, NullMetrics

    class MyComponent:
        def __init__(self, metrics: Metrics = NullMetrics()) -> None:
            self._m = metrics

        def do_work(self) -> None:
            self._m.increment("my_component.work_done", tags={"type": "heavy"})

**Usage (application side — Prometheus example):**

.. code-block:: python

    from prometheus_client import Counter, Histogram
    from RAW.utils.metrics import Metrics

    class PrometheusMetrics:
        _ops     = Counter("raw_operations_total", "...", ["name", "status"])
        _dur     = Histogram("raw_operation_duration_seconds", "...", ["name"])
        _counter = Counter("raw_counter_total", "...", ["name"])

        def increment(self, name, value=1.0, *, tags=None):
            label = (tags or {}).get("status", "")
            self._counter.labels(name=name, status=label).inc(value)

        def histogram(self, name, value, *, tags=None):
            self._dur.labels(name=name).observe(value)

        def gauge(self, name, value, *, tags=None):
            pass  # implement with prometheus_client.Gauge if needed

    # Wire in:
    storage = S3Storage(client=client, metrics=PrometheusMetrics())
    llm     = GeminiLLM(..., metrics=PrometheusMetrics())
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple, runtime_checkable


@runtime_checkable
class Metrics(Protocol):
    """Structural protocol for emitting metrics from any RAW component.

    Implementors need only satisfy the three method signatures — no base class
    or registration is required.  The :class:`NullMetrics` no-op is the
    default so components never need to guard against ``None``.

    **Tag conventions** (recommended, not enforced):

    +-------------------+-------------------------------+
    | Tag key           | Typical values                |
    +===================+===============================+
    | ``operation``     | ``"save"``, ``"open"``, …     |
    +-------------------+-------------------------------+
    | ``status``        | ``"success"``, ``"error"``    |
    +-------------------+-------------------------------+
    | ``component``     | ``"storage"``, ``"llm"``, …   |
    +-------------------+-------------------------------+
    | ``model``         | ``"gemini-2.0-flash"``, …     |
    +-------------------+-------------------------------+
    | ``direction``     | ``"upload"``, ``"download"``  |
    +-------------------+-------------------------------+
    """

    def increment(
        self,
        name: str,
        value: float = 1.0,
        *,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter metric by *value*.

        Args:
            name:  Dot-separated metric name, e.g. ``"raw.storage.operations"``.
            value: Amount to add.  Defaults to ``1.0``.
            tags:  Arbitrary key/value labels for the metric.
        """
        ...

    def histogram(
        self,
        name: str,
        value: float,
        *,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Record an observation in a histogram / distribution metric.

        Args:
            name:  Metric name, e.g. ``"raw.storage.operation_duration_seconds"``.
            value: Observed value (e.g. elapsed seconds, byte count).
            tags:  Arbitrary key/value labels.
        """
        ...

    def gauge(
        self,
        name: str,
        value: float,
        *,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge metric to an absolute *value*.

        Args:
            name:  Metric name, e.g. ``"raw.agent.queue_depth"``.
            value: Current absolute value.
            tags:  Arbitrary key/value labels.
        """
        ...


class NullMetrics:
    """No-op :class:`Metrics` implementation — the default for all components.

    Satisfies the :class:`Metrics` protocol without emitting anything.
    Use this when no metrics backend is configured, or in unit tests where
    metrics output is irrelevant.

    Example::

        assert isinstance(NullMetrics(), Metrics)  # True — satisfies protocol
    """

    __slots__ = ()

    def increment(
        self,
        name: str,
        value: float = 1.0,
        *,
        tags: dict[str, str] | None = None,
    ) -> None:
        pass

    def histogram(
        self,
        name: str,
        value: float,
        *,
        tags: dict[str, str] | None = None,
    ) -> None:
        pass

    def gauge(
        self,
        name: str,
        value: float,
        *,
        tags: dict[str, str] | None = None,
    ) -> None:
        pass


_Key = Tuple[str, Tuple[Tuple[str, str], ...]]


def _key(name: str, tags: dict[str, str] | None) -> _Key:
    return name, tuple(sorted((tags or {}).items()))


class InMemoryMetrics:
    """Collects metrics in memory, for tests and local debugging.

    ``counter("raw.tool.calls", tool="search")`` sums every counter series whose
    tags include the given ones; ``values`` and ``last`` read histograms and gauges.

    Example::

        metrics = InMemoryMetrics()
        agent = Agent(name="a", llm=llm, metrics=metrics)
        ...
        metrics.counter("raw.llm.calls", status="success")
    """

    def __init__(self) -> None:
        self.counters: Dict[_Key, float] = {}
        self.histograms: Dict[_Key, List[float]] = {}
        self.gauges: Dict[_Key, float] = {}

    def increment(self, name: str, value: float = 1.0, *, tags: dict[str, str] | None = None) -> None:
        key = _key(name, tags)
        self.counters[key] = self.counters.get(key, 0.0) + value

    def histogram(self, name: str, value: float, *, tags: dict[str, str] | None = None) -> None:
        self.histograms.setdefault(_key(name, tags), []).append(value)

    def gauge(self, name: str, value: float, *, tags: dict[str, str] | None = None) -> None:
        self.gauges[_key(name, tags)] = value

    @staticmethod
    def _matches(key: _Key, name: str, tags: Dict[str, str]) -> bool:
        return key[0] == name and all(item in key[1] for item in tags.items())

    def counter(self, name: str, **tags: str) -> float:
        return sum(v for k, v in self.counters.items() if self._matches(k, name, tags))

    def values(self, name: str, **tags: str) -> List[float]:
        return [x for k, v in self.histograms.items() if self._matches(k, name, tags) for x in v]

    def last(self, name: str, **tags: str) -> float | None:
        found = [v for k, v in self.gauges.items() if self._matches(k, name, tags)]
        return found[-1] if found else None


class OpenTelemetryMetrics:
    """Sends RAW metrics to OpenTelemetry. Needs ``pip install 'raw-agents[otel]'``.

    Uses the global meter provider, so configure an SDK and exporter in your
    application as usual; metric names and tags are passed through unchanged.
    """

    def __init__(self, meter_name: str = "raw") -> None:
        try:
            from opentelemetry import metrics as otel_metrics  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "OpenTelemetryMetrics requires OpenTelemetry. Install it with: pip install 'raw-agents[otel]'"
            ) from e
        self._meter = otel_metrics.get_meter(meter_name)
        self._instruments: Dict[Tuple[str, str], Any] = {}

    def _instrument(self, kind: str, name: str) -> Any:
        key = (kind, name)
        if key not in self._instruments:
            create = {
                "counter": self._meter.create_counter,
                "histogram": self._meter.create_histogram,
                "gauge": self._meter.create_gauge,
            }[kind]
            self._instruments[key] = create(name)
        return self._instruments[key]

    def increment(self, name: str, value: float = 1.0, *, tags: dict[str, str] | None = None) -> None:
        self._instrument("counter", name).add(value, attributes=tags or {})

    def histogram(self, name: str, value: float, *, tags: dict[str, str] | None = None) -> None:
        self._instrument("histogram", name).record(value, attributes=tags or {})

    def gauge(self, name: str, value: float, *, tags: dict[str, str] | None = None) -> None:
        self._instrument("gauge", name).set(value, attributes=tags or {})
