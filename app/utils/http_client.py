"""
Async HTTP Client
Replaces blocking requests with non-blocking httpx for async contexts
"""

import httpx
import certifi
from typing import Optional, Dict, Any

cafile = certifi.where()


class AsyncHTTPClient:
  """
  Async HTTP client using httpx.
  Singleton pattern untuk reuse connection pool.
  """

  _instance: Optional[httpx.AsyncClient] = None
  _instance_no_verify: Optional[httpx.AsyncClient] = None
  _timeout: float = 30.0

  @classmethod
  async def get_client(cls, verify: bool = False) -> httpx.AsyncClient:
    """Get or create async HTTP client instance."""
    if verify:
      if cls._instance is None:
        cls._instance = httpx.AsyncClient(
            timeout=cls._timeout,
            limits=httpx.Limits(
                max_connections=100, max_keepalive_connections=20
            ),
            verify=cafile,
        )
      return cls._instance
    else:
      if cls._instance_no_verify is None:
        cls._instance_no_verify = httpx.AsyncClient(
            timeout=cls._timeout,
            limits=httpx.Limits(
                max_connections=100, max_keepalive_connections=20
            ),
            verify=False,
        )
      return cls._instance_no_verify

  @classmethod
  async def close(cls):
    """Close the HTTP client."""
    if cls._instance:
      await cls._instance.aclose()
      cls._instance = None
    if cls._instance_no_verify:
      await cls._instance_no_verify.aclose()
      cls._instance_no_verify = None

  @classmethod
  async def post(
      cls,
      url: str,
      data: Optional[Dict[str, Any]] = None,
      json_data: Optional[Dict[str, Any]] = None,
      headers: Optional[Dict[str, str]] = None,
      timeout: Optional[float] = None,
      verify: bool = False,
  ) -> httpx.Response:
    """Make async POST request."""
    client = await cls.get_client(verify=verify)
    return await client.post(
        url,
        data=data,
        json=json_data,
        headers=headers,
        timeout=timeout or cls._timeout,
    )

  @classmethod
  async def get(
      cls,
      url: str,
      params: Optional[Dict[str, Any]] = None,
      headers: Optional[Dict[str, str]] = None,
      timeout: Optional[float] = None,
      verify: bool = False,
  ) -> httpx.Response:
    """Make async GET request."""
    client = await cls.get_client(verify=verify)
    return await client.get(
        url,
        params=params,
        headers=headers,
        timeout=timeout or cls._timeout,
    )


# Convenience functions
async def async_post(*args, **kwargs) -> httpx.Response:
  """Convenience function for async POST."""
  return await AsyncHTTPClient.post(*args, **kwargs)


async def async_get(*args, **kwargs) -> httpx.Response:
  """Convenience function for async GET."""
  return await AsyncHTTPClient.get(*args, **kwargs)
