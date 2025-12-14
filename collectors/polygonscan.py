"""Polygonscan blockchain explorer collector."""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from config.models import ExplorerConfig, RetryConfig
from collectors.base import ExplorerCollector
from collectors.models import Transaction, Holder, ActivityType


# Use standard logging to avoid circular imports
logger = logging.getLogger(__name__)


class PolygonscanCollector(ExplorerCollector):
    """Collector for Polygonscan API (Polygon/Matic blockchain).
    
    Implements data collection from Polygonscan's API for ERC-20 token
    transactions and holder information. The API is compatible with
    Etherscan's API format.
    """
    
    # Token decimals for stablecoins on Polygon
    # USDC on Polygon uses 6 decimals (bridged USDC.e)
    # USDT on Polygon uses 6 decimals
    TOKEN_DECIMALS = {
        "USDC": 6,
        "USDT": 6,
    }
    
    def __init__(
        self,
        config: ExplorerConfig,
        retry_config: Optional[RetryConfig] = None
    ):
        """Initialize the Polygonscan collector.
        
        Args:
            config: Explorer configuration with API details
            retry_config: Retry configuration (uses defaults if not provided)
        """
        super().__init__(config, retry_config)
        
        # Verify this is configured for Polygon
        if config.chain != "polygon":
            logger.warning(
                f"PolygonscanCollector initialized with chain '{config.chain}', expected 'polygon'"
            )
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse a Unix timestamp string to datetime.
        
        Args:
            timestamp_str: Unix timestamp as string
            
        Returns:
            datetime object in UTC
        """
        return datetime.fromtimestamp(int(timestamp_str), tz=timezone.utc)
    
    def _parse_amount(self, value_str: str, stablecoin: str) -> Decimal:
        """Parse a token amount from raw value to decimal.
        
        Args:
            value_str: Raw token value as string (in smallest unit)
            stablecoin: The stablecoin symbol for decimal lookup
            
        Returns:
            Decimal amount in standard units
        """
        decimals = self.TOKEN_DECIMALS.get(stablecoin, 18)
        raw_value = Decimal(value_str)
        return raw_value / Decimal(10 ** decimals)

    
    def _classify_activity(
        self,
        from_address: str,
        to_address: str,
        amount: Decimal
    ) -> ActivityType:
        """Classify the activity type of a transaction.
        
        Args:
            from_address: Sender address
            to_address: Receiver address
            amount: Transaction amount
            
        Returns:
            ActivityType classification
        """
        # Zero address indicates minting or burning
        zero_address = "0x0000000000000000000000000000000000000000"
        
        if from_address.lower() == zero_address:
            return ActivityType.OTHER  # Minting
        if to_address.lower() == zero_address:
            return ActivityType.OTHER  # Burning
        
        # Standard transfer
        if amount > 0 and from_address and to_address:
            return ActivityType.TRANSACTION
        
        return ActivityType.UNKNOWN
    
    def _parse_transaction(
        self,
        tx_data: dict,
        stablecoin: str
    ) -> Optional[Transaction]:
        """Parse a transaction from API response data.
        
        Args:
            tx_data: Raw transaction data from API
            stablecoin: The stablecoin symbol
            
        Returns:
            Transaction object or None if parsing fails
        """
        try:
            from_address = tx_data.get("from", "")
            to_address = tx_data.get("to", "")
            value = tx_data.get("value", "0")
            
            amount = self._parse_amount(value, stablecoin)
            activity_type = self._classify_activity(from_address, to_address, amount)
            
            # Parse gas information
            gas_used = None
            gas_price = None
            if tx_data.get("gasUsed"):
                gas_used = int(tx_data["gasUsed"])
            if tx_data.get("gasPrice"):
                gas_price = Decimal(tx_data["gasPrice"]) / Decimal(10 ** 9)  # Convert to Gwei
            
            return Transaction(
                transaction_hash=tx_data.get("hash", ""),
                block_number=int(tx_data.get("blockNumber", 0)),
                timestamp=self._parse_timestamp(tx_data.get("timeStamp", "0")),
                from_address=from_address,
                to_address=to_address,
                amount=amount,
                stablecoin=stablecoin,
                chain=self.chain,
                activity_type=activity_type,
                source_explorer=self.name,
                gas_used=gas_used,
                gas_price=gas_price,
            )
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(
                f"Failed to parse transaction from {self.name}: {e}",
                extra={
                    "explorer": self.name,
                    "tx_hash": tx_data.get("hash", "unknown"),
                    "error": str(e)
                }
            )
            return None
    
    async def fetch_stablecoin_transactions(
        self,
        stablecoin: str,
        contract_address: str,
        limit: int = 1000,
        run_id: Optional[str] = None
    ) -> list[Transaction]:
        """Fetch stablecoin transactions from Polygonscan.
        
        Uses the tokentx API endpoint to get ERC-20 token transfers.
        
        Args:
            stablecoin: The stablecoin symbol (e.g., "USDC", "USDT")
            contract_address: The token contract address
            limit: Maximum number of transactions to fetch
            run_id: Optional run ID for logging correlation
            
        Returns:
            List of Transaction objects
        """
        transactions: list[Transaction] = []
        
        log_extra = {
            "explorer": self.name,
            "stablecoin": stablecoin,
            "contract": contract_address,
        }
        if run_id:
            log_extra["run_id"] = run_id
        
        # Polygonscan API parameters (compatible with Etherscan format)
        params = {
            "module": "account",
            "action": "tokentx",
            "contractaddress": contract_address,
            "page": 1,
            "offset": min(limit, 10000),
            "sort": "desc",
        }
        
        logger.debug(
            f"Fetching {stablecoin} transactions from {self.name}",
            extra=log_extra
        )
        
        response = await self._make_request(params, run_id, endpoint="tokentx")
        
        if response is None:
            logger.error(
                f"Failed to fetch {stablecoin} transactions from {self.name}",
                extra=log_extra
            )
            return transactions
        
        result = response.get("result", [])
        
        # Handle case where result is an error message string
        if isinstance(result, str):
            if "No transactions found" in result:
                logger.info(
                    f"No {stablecoin} transactions found on {self.name}",
                    extra=log_extra
                )
            else:
                logger.warning(
                    f"Unexpected result from {self.name}: {result}",
                    extra=log_extra
                )
            return transactions
        
        # Parse transactions
        for tx_data in result[:limit]:
            tx = self._parse_transaction(tx_data, stablecoin)
            if tx:
                transactions.append(tx)
        
        logger.info(
            f"Parsed {len(transactions)} {stablecoin} transactions from {self.name}",
            extra={**log_extra, "count": len(transactions)}
        )
        
        return transactions

    
    def _parse_holder(
        self,
        holder_data: dict,
        stablecoin: str,
        contract_address: str
    ) -> Optional[Holder]:
        """Parse a holder from API response data.
        
        Args:
            holder_data: Raw holder data from API
            stablecoin: The stablecoin symbol
            contract_address: The token contract address
            
        Returns:
            Holder object or None if parsing fails
        """
        try:
            address = holder_data.get("TokenHolderAddress", "")
            balance_str = holder_data.get("TokenHolderQuantity", "0")
            
            balance = self._parse_amount(balance_str, stablecoin)
            
            # For holders, we don't have first_seen/last_activity from this endpoint
            now = datetime.now(timezone.utc)
            
            return Holder(
                address=address,
                balance=balance,
                stablecoin=stablecoin,
                chain=self.chain,
                first_seen=now,
                last_activity=now,
                is_store_of_value=False,  # Will be determined by ActivityClassifier
                source_explorer=self.name,
            )
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(
                f"Failed to parse holder from {self.name}: {e}",
                extra={
                    "explorer": self.name,
                    "address": holder_data.get("TokenHolderAddress", "unknown"),
                    "error": str(e)
                }
            )
            return None
    
    async def fetch_token_holders(
        self,
        stablecoin: str,
        contract_address: str,
        limit: int = 100,
        run_id: Optional[str] = None
    ) -> list[Holder]:
        """Fetch token holders from Polygonscan.
        
        Note: The token holder list endpoint may require a Pro API key.
        
        Args:
            stablecoin: The stablecoin symbol (e.g., "USDC", "USDT")
            contract_address: The token contract address
            limit: Maximum number of holders to fetch
            run_id: Optional run ID for logging correlation
            
        Returns:
            List of Holder objects
        """
        holders: list[Holder] = []
        
        log_extra = {
            "explorer": self.name,
            "stablecoin": stablecoin,
            "contract": contract_address,
        }
        if run_id:
            log_extra["run_id"] = run_id
        
        # Token holder list endpoint
        params = {
            "module": "token",
            "action": "tokenholderlist",
            "contractaddress": contract_address,
            "page": 1,
            "offset": min(limit, 1000),
        }
        
        logger.debug(
            f"Fetching {stablecoin} holders from {self.name}",
            extra=log_extra
        )
        
        response = await self._make_request(
            params, run_id, endpoint="tokenholderlist"
        )
        
        if response is None:
            logger.warning(
                f"Failed to fetch {stablecoin} holders from {self.name}",
                extra=log_extra
            )
            return holders
        
        result = response.get("result", [])
        
        # Handle case where result is an error message string
        if isinstance(result, str):
            if "No token holder found" in result:
                logger.info(
                    f"No {stablecoin} holders found on {self.name}",
                    extra=log_extra
                )
            elif "API Pro" in result or "upgrade" in result.lower():
                logger.info(
                    f"Token holder endpoint requires Pro API on {self.name}",
                    extra=log_extra
                )
            else:
                logger.warning(
                    f"Unexpected result from {self.name}: {result}",
                    extra=log_extra
                )
            return holders
        
        # Parse holders
        for holder_data in result[:limit]:
            holder = self._parse_holder(holder_data, stablecoin, contract_address)
            if holder:
                holders.append(holder)
        
        logger.info(
            f"Parsed {len(holders)} {stablecoin} holders from {self.name}",
            extra={**log_extra, "count": len(holders)}
        )
        
        return holders
