"""
Retry utilities for YouTube fetcher with exponential backoff.

This module provides retry logic specifically tuned for YouTube API operations,
including exponential backoff to handle rate limiting gracefully.
"""

import time
import logging
from typing import Callable, TypeVar, Optional

T = TypeVar("T")


def execute_with_exponential_backoff(
    func: Callable[[], T],
    description: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> T:
    """
    Execute a function with exponential backoff retry logic.

    This is useful for handling rate limiting and temporary network issues
    when fetching data from YouTube.

    Args:
        func: Function to execute (should take no arguments)
        description: Description of the operation for logging
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)

    Returns:
        Result of the function

    Raises:
        Exception: The last exception if all retries fail

    Example:
        >>> def fetch_data():
        ...     return api.get_video("abc123")
        >>> result = execute_with_exponential_backoff(
        ...     fetch_data,
        ...     "Fetching video abc123",
        ...     max_retries=3
        ... )
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e

            if attempt == max_retries - 1:
                # Last attempt failed
                logging.error(
                    f"Failed after {max_retries} attempts: {description}. Error: {e}"
                )
                raise

            # Calculate delay with exponential backoff
            delay = min(base_delay * (2 ** attempt), max_delay)

            logging.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {description}. "
                f"Retrying in {delay:.1f}s... Error: {e}"
            )

            time.sleep(delay)

    # Should never reach here, but just in case
    raise last_exception


def execute_with_retry_and_skip(
    func: Callable[[], T],
    description: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Optional[T]:
    """
    Execute a function with retry logic, returning None instead of raising on failure.

    This is useful when processing multiple items where individual failures
    should be logged but not stop the entire process.

    Args:
        func: Function to execute
        description: Description of the operation for logging
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds (default: 1.0)

    Returns:
        Result of the function, or None if all retries failed

    Example:
        >>> def fetch_transcript(video_id):
        ...     return api.get_transcript(video_id)
        >>> transcript = execute_with_retry_and_skip(
        ...     lambda: fetch_transcript("abc123"),
        ...     "Fetching transcript for abc123"
        ... )
        >>> if transcript is None:
        ...     print("Failed to fetch transcript")
    """
    try:
        return execute_with_exponential_backoff(
            func,
            description,
            max_retries=max_retries,
            base_delay=base_delay,
        )
    except Exception as e:
        logging.error(f"All retries failed for: {description}. Skipping. Error: {e}")
        return None
