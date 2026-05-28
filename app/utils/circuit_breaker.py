"""
Circuit Breaker Pattern untuk external API calls.
Mencegah cascade failures saat external API down.
"""

import time
import asyncio
from enum import Enum
from typing import Callable, Optional, Any
from dataclasses import dataclass
from functools import wraps

from app.logger import Logger


class CircuitState(Enum):
  """States untuk circuit breaker."""

  CLOSED = "closed"  # Normal operation
  OPEN = "open"  # Failing, reject calls
  HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
  """Configuration untuk circuit breaker."""

  failure_threshold: int = 5  # Failures sebelum open
  success_threshold: int = 3  # Successes sebelum close dari half-open
  cooldown_seconds: float = 60.0  # Waktu sebelum half-open
  timeout_seconds: float = 30.0  # Timeout untuk API calls


class CircuitBreakerError(Exception):
  """Exception ketika circuit breaker is OPEN."""

  def __init__(self, message: str = "Circuit breaker is OPEN"):
    self.message = message
    super().__init__(self.message)


class CircuitBreaker:
  """
  Circuit Breaker untuk melindungi dari external API failures.

  States:
  - CLOSED: Normal operation, pass calls through
  - OPEN: Too many failures, reject calls immediately
  - HALF_OPEN: Testing if service recovered

  Usage:
      # Direct usage
      cb = CircuitBreaker("external_api")

      @cb.protect
      async def call_external_api():
          # Your API call here
          pass

      # Or manual check
      if cb.can_execute():
          try:
              result = await api_call()
              cb.record_success()
          except Exception:
              cb.record_failure()
  """

  def __init__(
      self,
      name: str,
      config: Optional[CircuitBreakerConfig] = None,
      fallback: Optional[Callable] = None,
  ):
    self.name = name
    self.config = config or CircuitBreakerConfig()
    self.fallback = fallback
    self.logger = Logger(f"circuit_breaker_{name}")

    # State
    self._state = CircuitState.CLOSED
    self._failure_count = 0
    self._success_count = 0
    self._last_failure_time: Optional[float] = None

    # Statistics
    self._stats = {
        "total_calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "rejected_calls": 0,
        "state_changes": 0,
    }

  @property
  def state(self) -> CircuitState:
    """Get current state."""
    return self._state

  def can_execute(self) -> bool:
    """
    Check if call can be executed.

    Returns:
        bool: True if circuit allows execution
    """
    self._stats["total_calls"] += 1

    if self._state == CircuitState.CLOSED:
      return True

    if self._state == CircuitState.OPEN:
      # Check if should try half-open
      if self._should_attempt_reset():
        self._transition_to(CircuitState.HALF_OPEN)
        return True
      else:
        self._stats["rejected_calls"] += 1
        return False

    if self._state == CircuitState.HALF_OPEN:
      return True

    return False

  def _should_attempt_reset(self) -> bool:
    """Check if enough time passed to try reset."""
    if self._last_failure_time is None:
      return True

    elapsed = time.time() - self._last_failure_time
    return elapsed >= self.config.cooldown_seconds

  def record_success(self):
    """Record successful call."""
    self._failure_count = 0
    self._stats["successful_calls"] += 1

    if self._state == CircuitState.HALF_OPEN:
      self._success_count += 1
      if self._success_count >= self.config.success_threshold:
        self._transition_to(CircuitState.CLOSED)
        self._success_count = 0

  def record_failure(self):
    """Record failed call."""
    self._failure_count += 1
    self._last_failure_time = time.time()
    self._stats["failed_calls"] += 1

    if self._state == CircuitState.CLOSED:
      if self._failure_count >= self.config.failure_threshold:
        self._transition_to(CircuitState.OPEN)

    elif self._state == CircuitState.HALF_OPEN:
      # Immediately open again
      self._transition_to(CircuitState.OPEN)
      self._success_count = 0

  def _transition_to(self, new_state: CircuitState):
    """Transition to new state."""
    old_state = self._state
    self._state = new_state
    self._stats["state_changes"] += 1

    self.logger.warning(
        f"Circuit breaker '{self.name}' transitioned: "
        f"{old_state.value} -> {new_state.value}"
    )

  async def execute(
      self, func: Callable, *args, fallback: Optional[Callable] = None, **kwargs
  ) -> Any:
    """
    Execute function dengan circuit breaker protection.

    Args:
        func: Function to execute
        fallback: Optional fallback function jika circuit open
        *args, **kwargs: Arguments untuk func

    Returns:
        Result from func or fallback

    Raises:
        CircuitBreakerError: If circuit is OPEN and no fallback
    """
    if not self.can_execute():
      if fallback:
        self.logger.debug(f"Using fallback for {func.__name__}")
        return (
            await fallback(*args, **kwargs)
            if asyncio.iscoroutinefunction(fallback)
            else fallback(*args, **kwargs)
        )
      elif self.fallback:
        return (
            await self.fallback(*args, **kwargs)
            if asyncio.iscoroutinefunction(self.fallback)
            else self.fallback(*args, **kwargs)
        )
      else:
        raise CircuitBreakerError(
            f"Circuit breaker '{self.name}' is OPEN and no fallback provided"
        )

    try:
      # Execute dengan timeout
      if asyncio.iscoroutinefunction(func):
        result = await asyncio.wait_for(
            func(*args, **kwargs), timeout=self.config.timeout_seconds
        )
      else:
        result = func(*args, **kwargs)

      self.record_success()
      return result

    except asyncio.TimeoutError:
      self.record_failure()
      raise
    except Exception as e:
      self.record_failure()
      raise

  def protect(self, fallback: Optional[Callable] = None):
    """
    Decorator untuk protect function dengan circuit breaker.

    Usage:
        cb = CircuitBreaker("api")

        @cb.protect(fallback=default_value)
        async def call_api():
            # API call
            pass
    """

    def decorator(func: Callable):
      @wraps(func)
      async def async_wrapper(*args, **kwargs):
        return await self.execute(func, *args, fallback=fallback, **kwargs)

      @wraps(func)
      def sync_wrapper(*args, **kwargs):
        # For sync functions, use run_in_executor
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.execute(func, *args, fallback=fallback, **kwargs)
        )

      return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

  def get_stats(self) -> dict:
    """Get circuit breaker statistics."""
    return {
        **self._stats,
        "current_state": self._state.value,
        "failure_count": self._failure_count,
        "success_count": self._success_count,
        "last_failure": self._last_failure_time,
    }

  def reset(self):
    """Manual reset ke CLOSED state."""
    self._state = CircuitState.CLOSED
    self._failure_count = 0
    self._success_count = 0
    self._last_failure_time = None
    self.logger.info(f"Circuit breaker '{self.name}' manually reset")


# Registry untuk circuit breakers
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str) -> CircuitBreaker:
  """Get or create circuit breaker."""
  if name not in _circuit_breakers:
    _circuit_breakers[name] = CircuitBreaker(name)
  return _circuit_breakers[name]


def get_all_circuit_breakers() -> dict[str, CircuitBreaker]:
  """Get all circuit breakers."""
  return _circuit_breakers.copy()


def get_all_stats() -> dict[str, dict]:
  """Get stats from all circuit breakers."""
  return {name: cb.get_stats() for name, cb in _circuit_breakers.items()}
