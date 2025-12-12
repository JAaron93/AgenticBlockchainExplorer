"""Base class for blockchain explorer collectors."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import aiohttp

from config.models import ExplorerConfig, RetryConfig
from collectors.models import Transaction, Holder, ExplorerData


logger = logging.getLogger(__name__)


class ExplorerCollector(ABC):
    """Abstract base class for blockchain explorer data collectors.
    
    Provides common functionality for rate limiting, retry logic,
    and response validation. Concrete implementations must implement
    the fetch_stablecoin_transactions and fetch_token_holders methods.
    """
    
    def __init__(
        self,
        config: ExplorerConfig,
        retry_config: Optional[RetryConfig] = None
    ):
        """Initialize the collector.
        
        Args:
            config: Explorer configuration with API details
            retry_config: Retry configuration (uses defaults if not provided)
        """
        self.config = config
        self.retry_config = retry_config or RetryConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._request_count = 0
        self._last_request_time: Optional[float] = None
    
    @property
    def name(self) -> str:
        """Get the explorer name."""
        return self.config.name
    
    @property
    def chain(self) -> str:
        """Get the blockchain chain."""
        return self.config.chain
    
    @property
    def base_url(self) -> str:
        """Get the base API URL."""
        return str(self.config.base_url)
    
    @property
    def api_key(self) -> str:
        """Get the API key."""
        return self.config.api_key

    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self.retry_config.request_timeout_seconds
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def __aenter__(self) -> "ExplorerCollector":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def handle_rate_limit(self) -> None:
        """Handle rate limiting by waiting before the next request.
        
        Implements a simple rate limiting strategy by waiting
        the configured backoff time when rate limited.
        """
        wait_time = self.retry_config.backoff_seconds
        logger.warning(
            f"Rate limited on {self.name}. Waiting {wait_time} seconds before retry.",
            extra={"explorer": self.name, "wait_seconds": wait_time}
        )
        await asyncio.sleep(wait_time)
    
    def validate_response(self, response: dict) -> bool:
        """Validate an API response.
        
        Args:
            response: The JSON response from the API
            
        Returns:
            True if the response is valid, False otherwise
        """
        if not isinstance(response, dict):
            logger.error(
                f"Invalid response type from {self.name}: expected dict, got {type(response)}",
                extra={"explorer": self.name}
            )
            return False
        
        # Check for common error indicators
        status = response.get("status")
        message = response.get("message", "")
        
        # Status "0" typically indicates an error in *scan APIs
        if status == "0":
            # "No transactions found" is not an error, just empty results
            if "No transactions found" in message or "No records found" in message:
                return True
            logger.warning(
                f"API error from {self.name}: {message}",
                extra={"explorer": self.name, "message": message}
            )
            return False
        
        return True
    
    def _is_rate_limit_error(self, response: dict) -> bool:
        """Check if the response indicates a rate limit error.
        
        Args:
            response: The JSON response from the API
            
        Returns:
            True if rate limited, False otherwise
        """
        message = response.get("message", "").lower()
        result = response.get("result", "")
        
        rate_limit_indicators = [
            "rate limit",
            "max rate limit",
            "too many requests",
            "exceeded the rate limit",
        ]
        
        for indicator in rate_limit_indicators:
            if indicator in message.lower() or (isinstance(result, str) and indicator in result.lower()):
                return True
        
        return False

    
    async def _make_request(
        self,
        params: dict[str, Any],
        run_id: Optional[str] = None
    ) -> Optional[dict]:
        """Make an API request with retry logic and exponential backoff.
        
        Args:
            params: Query parameters for the API request
            run_id: Optional run ID for logging correlation
            
        Returns:
            The JSON response if successful, None otherwise
        """
        session = await self._get_session()
        
        # Add API key to params
        params["apikey"] = self.api_key
        
        last_error: Optional[Exception] = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                log_extra = {
                    "explorer": self.name,
                    "attempt": attempt + 1,
                    "max_attempts": self.retry_config.max_attempts,
                }
                if run_id:
                    log_extra["run_id"] = run_id
                
                logger.debug(
                    f"Making request to {self.name} (attempt {attempt + 1}/{self.retry_config.max_attempts})",
                    extra=log_extra
                )
                
                async with session.get(self.base_url, params=params) as response:
                    # Handle HTTP-level rate limiting
                    if response.status == 429:
                        logger.warning(
                            f"HTTP 429 rate limit from {self.name}",
                            extra=log_extra
                        )
                        await self.handle_rate_limit()
                        continue
                    
                    # Handle other HTTP errors
                    if response.status != 200:
                        logger.error(
                            f"HTTP {response.status} from {self.name}",
                            extra={**log_extra, "status_code": response.status}
                        )
                        if attempt < self.retry_config.max_attempts - 1:
                            backoff = self.retry_config.backoff_seconds * (2 ** attempt)
                            await asyncio.sleep(backoff)
                            continue
                        return None
                    
                    data = await response.json()
                    
                    # Check for API-level rate limiting
                    if self._is_rate_limit_error(data):
                        await self.handle_rate_limit()
                        continue
                    
                    # Validate response
                    if not self.validate_response(data):
                        if attempt < self.retry_config.max_attempts - 1:
                            backoff = self.retry_config.backoff_seconds * (2 ** attempt)
                            await asyncio.sleep(backoff)
                            continue
                        return None
                    
                    return data
                    
            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError(f"Request to {self.name} timed out")
                logger.warning(
                    f"Request timeout to {self.name} (attempt {attempt + 1})",
                    extra=log_extra
                )
            except aiohttp.ClientError as e:
                last_error = e
                logger.warning(
                    f"Network error from {self.name}: {e} (attempt {attempt + 1})",
                    extra={**log_extra, "error": str(e)}
                )
            except Exception as e:
                last_error = e
                logger.error(
                    f"Unexpected error from {self.name}: {e}",
                    extra={**log_extra, "error": str(e)},
                    exc_info=True
                )
            
            # Exponential backoff before retry
            if attempt < self.retry_config.max_attempts - 1:
                backoff = self.retry_config.backoff_seconds * (2 ** attempt)
                logger.info(
                    f"Retrying {self.name} in {backoff} seconds",
                    extra={**log_extra, "backoff_seconds": backoff}
                )
                await asyncio.sleep(backoff)
        
        logger.error(
            f"All {self.retry_config.max_attempts} attempts failed for {self.name}",
            extra={"explorer": self.name, "last_error": str(last_error)}
        )
        return None

    
    @abstractmethod
    async def fetch_stablecoin_transactions(
        self,
        stablecoin: str,
        contract_address: str,
        limit: int = 1000,
        run_id: Optional[str] = None
    ) -> list[Transaction]:
        """Fetch stablecoin transactions from the explorer.
        
        Args:
            stablecoin: The stablecoin symbol (e.g., "USDC", "USDT")
            contract_address: The token contract address
            limit: Maximum number of transactions to fetch
            run_id: Optional run ID for logging correlation
            
        Returns:
            List of Transaction objects
        """
        pass
    
    @abstractmethod
    async def fetch_token_holders(
        self,
        stablecoin: str,
        contract_address: str,
        limit: int = 100,
        run_id: Optional[str] = None
    ) -> list[Holder]:
        """Fetch token holders from the explorer.
        
        Args:
            stablecoin: The stablecoin symbol (e.g., "USDC", "USDT")
            contract_address: The token contract address
            limit: Maximum number of holders to fetch
            run_id: Optional run ID for logging correlation
            
        Returns:
            List of Holder objects
        """
        pass
    
    async def collect_all(
        self,
        stablecoins: dict[str, str],
        max_records: int = 1000,
        run_id: Optional[str] = None
    ) -> ExplorerData:
        """Collect all data from this explorer for the given stablecoins.
        
        Args:
            stablecoins: Dict mapping stablecoin symbol to contract address
            max_records: Maximum records per stablecoin
            run_id: Optional run ID for logging correlation
            
        Returns:
            ExplorerData containing all collected transactions and holders
        """
        import time
        start_time = time.time()
        
        result = ExplorerData(
            explorer_name=self.name,
            chain=self.chain
        )
        
        log_extra = {"explorer": self.name, "chain": self.chain}
        if run_id:
            log_extra["run_id"] = run_id
        
        logger.info(
            f"Starting data collection from {self.name} for {len(stablecoins)} stablecoins",
            extra=log_extra
        )
        
        for stablecoin, contract_address in stablecoins.items():
            try:
                logger.info(
                    f"Fetching {stablecoin} transactions from {self.name}",
                    extra={**log_extra, "stablecoin": stablecoin}
                )
                
                transactions = await self.fetch_stablecoin_transactions(
                    stablecoin=stablecoin,
                    contract_address=contract_address,
                    limit=max_records,
                    run_id=run_id
                )
                result.transactions.extend(transactions)
                
                logger.info(
                    f"Fetched {len(transactions)} {stablecoin} transactions from {self.name}",
                    extra={**log_extra, "stablecoin": stablecoin, "count": len(transactions)}
                )
                
            except Exception as e:
                error_msg = f"Error fetching {stablecoin} transactions: {e}"
                result.errors.append(error_msg)
                logger.error(error_msg, extra=log_extra, exc_info=True)
            
            try:
                logger.info(
                    f"Fetching {stablecoin} holders from {self.name}",
                    extra={**log_extra, "stablecoin": stablecoin}
                )
                
                holders = await self.fetch_token_holders(
                    stablecoin=stablecoin,
                    contract_address=contract_address,
                    limit=min(max_records, 100),  # Holders typically limited
                    run_id=run_id
                )
                result.holders.extend(holders)
                
                logger.info(
                    f"Fetched {len(holders)} {stablecoin} holders from {self.name}",
                    extra={**log_extra, "stablecoin": stablecoin, "count": len(holders)}
                )
                
            except Exception as e:
                error_msg = f"Error fetching {stablecoin} holders: {e}"
                result.errors.append(error_msg)
                logger.error(error_msg, extra=log_extra, exc_info=True)
        
        result.collection_time_seconds = time.time() - start_time
        
        logger.info(
            f"Completed collection from {self.name}: {result.total_records} records in {result.collection_time_seconds:.2f}s",
            extra={
                **log_extra,
                "total_records": result.total_records,
                "transactions": len(result.transactions),
                "holders": len(result.holders),
                "errors": len(result.errors),
                "duration_seconds": result.collection_time_seconds
            }
        )
        
        return result
