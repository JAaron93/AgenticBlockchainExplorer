import re
import sys

def modify_file(filepath, callback):
    with open(filepath, 'r') as f:
        content = f.read()
    new_content = callback(content)
    with open(filepath, 'w') as f:
        f.write(new_content)

def refactor_base(content):
    # Add imports
    content = re.sub(
        r'from typing import Any, Optional',
        r'from datetime import datetime, timezone\nfrom decimal import Decimal\nfrom typing import Any, Optional',
        content
    )
    content = content.replace(
        'from collectors.models import Transaction, Holder, ExplorerData',
        'from collectors.models import Transaction, Holder, ExplorerData, ActivityType'
    )
    
    # Add _get_blockchain_validator
    validator_code = """
# Lazy import for blockchain validator to avoid circular imports
_blockchain_validator = None

def _get_blockchain_validator():
    \"\"\"Get or create the blockchain validator singleton.\"\"\"
    global _blockchain_validator
    if _blockchain_validator is None:
        try:
            from core.security.blockchain_validator import BlockchainDataValidator
            _blockchain_validator = BlockchainDataValidator()
            logger.debug("BlockchainDataValidator initialized for ExplorerCollector")
        except Exception as e:
            logger.warning(f"Failed to initialize BlockchainDataValidator: {e}")
            _blockchain_validator = False  # Mark as failed, don't retry
    return _blockchain_validator if _blockchain_validator else None
"""
    content = content.replace(
        '# Lazy import for schema validator to avoid circular imports',
        validator_code.strip() + '\n\n# Lazy import for schema validator to avoid circular imports'
    )
    
    # Define TOKEN_DECIMALS
    content = content.replace(
        'class ExplorerCollector(ABC):',
        'class ExplorerCollector(ABC):\n    # Token decimals (override in subclasses)\n    TOKEN_DECIMALS: dict[str, int] = {}'
    )
    
    # The concrete methods to add
    concrete_methods = """
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        \"\"\"Parse a Unix timestamp string to datetime.
        
        Args:
            timestamp_str: Unix timestamp as string
            
        Returns:
            datetime object in UTC
        \"\"\"
        return datetime.fromtimestamp(int(timestamp_str), tz=timezone.utc)
    
    def _parse_amount(self, value_str: str, stablecoin: str) -> Decimal:
        \"\"\"Parse a token amount from raw value to decimal.
        
        Args:
            value_str: Raw token value as string (in smallest unit)
            stablecoin: The stablecoin symbol for decimal lookup
            
        Returns:
            Decimal amount in standard units
        \"\"\"
        decimals = self.TOKEN_DECIMALS.get(stablecoin, 18)
        raw_value = Decimal(value_str)
        return raw_value / Decimal(10 ** decimals)

    def _classify_activity(
        self,
        from_address: str,
        to_address: str,
        amount: Decimal
    ) -> ActivityType:
        \"\"\"Classify the activity type of a transaction.
        
        Args:
            from_address: Sender address
            to_address: Receiver address
            amount: Transaction amount
            
        Returns:
            ActivityType classification
        \"\"\"
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
        \"\"\"Parse a transaction from API response data.
        
        Args:
            tx_data: Raw transaction data from API
            stablecoin: The stablecoin symbol
            
        Returns:
            Transaction object or None if parsing/validation fails
        \"\"\"
        try:
            from_address = tx_data.get("from", "")
            to_address = tx_data.get("to", "")
            tx_hash = tx_data.get("hash", "")
            value = tx_data.get("value", "0")
            
            # Validate using BlockchainDataValidator if available
            validator = _get_blockchain_validator()
            if validator:
                if tx_hash and not validator.validate_tx_hash(tx_hash):
                    logger.warning(
                        f"Skipping record with invalid field: tx_hash",
                        extra={"explorer": self.name, "field": "tx_hash"}
                    )
                    return None
                
                if from_address and not validator.validate_address(from_address):
                    logger.warning(
                        f"Skipping record with invalid field: from_address",
                        extra={"explorer": self.name, "field": "from_address"}
                    )
                    return None
                
                if to_address and not validator.validate_address(to_address):
                    logger.warning(
                        f"Skipping record with invalid field: to_address",
                        extra={"explorer": self.name, "field": "to_address"}
                    )
                    return None
                
                from_address = validator.normalize_address(from_address)
                to_address = validator.normalize_address(to_address)
            
            amount = self._parse_amount(value, stablecoin)
            activity_type = self._classify_activity(from_address, to_address, amount)
            
            gas_used = None
            gas_price = None
            if tx_data.get("gasUsed"):
                gas_used = int(tx_data["gasUsed"])
            if tx_data.get("gasPrice"):
                gas_price = Decimal(tx_data["gasPrice"]) / Decimal(10 ** 9)
            
            return Transaction(
                transaction_hash=tx_hash,
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
                extra={"explorer": self.name, "error": str(e)}
            )
            return None

    async def fetch_stablecoin_transactions(
        self,
        stablecoin: str,
        contract_address: str,
        limit: int = 1000,
        run_id: Optional[str] = None
    ) -> list[Transaction]:
        \"\"\"Fetch stablecoin transactions from the explorer.\"\"\"
        transactions: list[Transaction] = []
        
        log_extra = {
            "explorer": self.name,
            "stablecoin": stablecoin,
            "contract": contract_address,
        }
        if run_id:
            log_extra["run_id"] = run_id
        
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
            logger.error(f"Failed to fetch {stablecoin} transactions from {self.name}", extra=log_extra)
            return transactions
        
        result = response.get("result", [])
        
        if isinstance(result, str):
            if "No transactions found" in result:
                logger.info(f"No {stablecoin} transactions found on {self.name}", extra=log_extra)
            else:
                logger.warning(f"Unexpected result from {self.name}: {result}", extra=log_extra)
            return transactions
        
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
        \"\"\"Parse a holder from API response data.\"\"\"
        try:
            address = holder_data.get("TokenHolderAddress", "")
            balance_str = holder_data.get("TokenHolderQuantity", "0")
            
            validator = _get_blockchain_validator()
            if validator:
                if address and not validator.validate_address(address):
                    logger.warning(
                        f"Skipping record with invalid field: address",
                        extra={"explorer": self.name, "field": "address"}
                    )
                    return None
                address = validator.normalize_address(address)
            
            balance = self._parse_amount(balance_str, stablecoin)
            now = datetime.now(timezone.utc)
            
            return Holder(
                address=address,
                balance=balance,
                stablecoin=stablecoin,
                chain=self.chain,
                first_seen=now,
                last_activity=now,
                is_store_of_value=False,
                source_explorer=self.name,
            )
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(
                f"Failed to parse holder from {self.name}: {e}",
                extra={"explorer": self.name, "error": str(e)}
            )
            return None
    
    async def fetch_token_holders(
        self,
        stablecoin: str,
        contract_address: str,
        limit: int = 100,
        run_id: Optional[str] = None
    ) -> list[Holder]:
        \"\"\"Fetch token holders from the explorer.\"\"\"
        holders: list[Holder] = []
        
        log_extra = {
            "explorer": self.name,
            "stablecoin": stablecoin,
            "contract": contract_address,
        }
        if run_id:
            log_extra["run_id"] = run_id
        
        params = {
            "module": "token",
            "action": "tokenholderlist",
            "contractaddress": contract_address,
            "page": 1,
            "offset": min(limit, 1000),
        }
        
        response = await self._make_request(
            params, run_id, endpoint="tokenholderlist"
        )
        
        if response is None:
            logger.warning(f"Failed to fetch {stablecoin} holders from {self.name}", extra=log_extra)
            return holders
        
        result = response.get("result", [])
        
        if isinstance(result, str):
            if "No token holder found" in result:
                logger.info(f"No {stablecoin} holders found on {self.name}", extra=log_extra)
            elif "API Pro" in result or "upgrade" in result.lower():
                logger.info(f"Token holder endpoint requires Pro API on {self.name}", extra=log_extra)
            else:
                logger.warning(f"Unexpected result from {self.name}: {result}", extra=log_extra)
            return holders
        
        for holder_data in result[:limit]:
            holder = self._parse_holder(holder_data, stablecoin, contract_address)
            if holder:
                holders.append(holder)
        
        logger.info(
            f"Parsed {len(holders)} {stablecoin} holders from {self.name}",
            extra={**log_extra, "count": len(holders)}
        )
        
        return holders
"""
    
    # Replace the abstract methods with concrete implementations
    pattern = r'    @abstractmethod\n    async def fetch_stablecoin_transactions\(.*?pass\n\s*@abstractmethod\n    async def fetch_token_holders\(.*?pass'
    content = re.sub(pattern, concrete_methods.strip('\n'), content, flags=re.DOTALL)
    
    return content

def refactor_subclass(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Remove lazy import of _blockchain_validator (from line: # Lazy import for blockchain validator ... down to ... Mark as failed...)
    # We can match exactly the pattern up to 'return _blockchain_validator...'
    # Or just use the class start
    # Let's match from `# Lazy import for blockchain validator` until `class`
    content = re.sub(r'# Lazy import for blockchain validator.*?def _get_blockchain_validator\(\):.*?(?=\n\n\nclass \w+Collector)', '', content, flags=re.DOTALL)
    
    idx = content.find('def _parse_timestamp(')
    if idx != -1:
        new_content = content[:idx-4]
        with open(filepath, 'w') as f:
            f.write(new_content.rstrip() + '\\n')
    else:
        print(f"Warning: _parse_timestamp not found in {filepath}")

modify_file('/Users/pretermodernist/AgenticBlockchainExplorer/collectors/base.py', refactor_base)
refactor_subclass('/Users/pretermodernist/AgenticBlockchainExplorer/collectors/etherscan.py')
refactor_subclass('/Users/pretermodernist/AgenticBlockchainExplorer/collectors/bscscan.py')
refactor_subclass('/Users/pretermodernist/AgenticBlockchainExplorer/collectors/polygonscan.py')
print("Done refactoring collectors.")
