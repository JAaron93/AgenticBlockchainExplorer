"""Integration tests for collectors with mock API responses."""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collectors.etherscan import EtherscanCollector
from collectors.bscscan import BscscanCollector
from collectors.polygonscan import PolygonscanCollector
from collectors.models import ActivityType, Transaction, Holder
from config.models import ExplorerConfig, RetryConfig


@pytest.fixture
def etherscan_config():
    """Create Etherscan explorer configuration."""
    return ExplorerConfig(
        name="etherscan",
        base_url="https://api.etherscan.io/api",
        api_key="test_api_key",
        type="api",
        chain="ethereum",
    )


@pytest.fixture
def bscscan_config():
    """Create BscScan explorer configuration."""
    return ExplorerConfig(
        name="bscscan",
        base_url="https://api.bscscan.com/api",
        api_key="test_api_key",
        type="api",
        chain="bsc",
    )


@pytest.fixture
def polygonscan_config():
    """Create Polygonscan explorer configuration."""
    return ExplorerConfig(
        name="polygonscan",
        base_url="https://api.polygonscan.com/api",
        api_key="test_api_key",
        type="api",
        chain="polygon",
    )


@pytest.fixture
def retry_config():
    """Create retry configuration for tests."""
    return RetryConfig(
        max_attempts=2,
        backoff_seconds=1,  # Minimum for tests
        request_timeout_seconds=5,
    )


@pytest.fixture
def mock_transaction_response():
    """Create a mock API response for transactions."""
    return {
        "status": "1",
        "message": "OK",
        "result": [
            {
                "hash": "0x123abc456def",
                "blockNumber": "18500000",
                "timeStamp": "1699000000",
                "from": "0x1111111111111111111111111111111111111111",
                "to": "0x2222222222222222222222222222222222222222",
                "value": "1000000000",  # 1000 USDC (6 decimals)
                "gasUsed": "65000",
                "gasPrice": "20000000000",
            },
            {
                "hash": "0x789ghi012jkl",
                "blockNumber": "18500001",
                "timeStamp": "1699000100",
                "from": "0x3333333333333333333333333333333333333333",
                "to": "0x4444444444444444444444444444444444444444",
                "value": "500000000",  # 500 USDC
                "gasUsed": "55000",
                "gasPrice": "18000000000",
            },
        ],
    }


@pytest.fixture
def mock_holder_response():
    """Create a mock API response for holders."""
    return {
        "status": "1",
        "message": "OK",
        "result": [
            {
                "TokenHolderAddress": "0xaaaa111111111111111111111111111111111111",
                "TokenHolderQuantity": "50000000000",  # 50000 USDC
            },
            {
                "TokenHolderAddress": "0xbbbb222222222222222222222222222222222222",
                "TokenHolderQuantity": "25000000000",  # 25000 USDC
            },
        ],
    }


@pytest.fixture
def mock_empty_response():
    """Create a mock API response with no results."""
    return {
        "status": "0",
        "message": "No transactions found",
        "result": [],
    }


@pytest.fixture
def mock_rate_limit_response():
    """Create a mock rate limit response."""
    return {
        "status": "0",
        "message": "Max rate limit reached",
        "result": "Max rate limit reached, please use API Key for higher rate limit",
    }


class TestEtherscanCollector:
    """Integration tests for EtherscanCollector."""

    @pytest.mark.asyncio
    async def test_fetch_transactions_success(
        self, etherscan_config, retry_config, mock_transaction_response
    ):
        """Successfully fetch and parse transactions."""
        collector = EtherscanCollector(etherscan_config, retry_config)

        with patch.object(
            collector, "_make_request", return_value=mock_transaction_response
        ):
            transactions = await collector.fetch_stablecoin_transactions(
                stablecoin="USDC",
                contract_address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                limit=100,
            )

            assert len(transactions) == 2
            assert all(isinstance(tx, Transaction) for tx in transactions)

            # Check first transaction
            tx1 = transactions[0]
            assert tx1.transaction_hash == "0x123abc456def"
            assert tx1.stablecoin == "USDC"
            assert tx1.chain == "ethereum"
            assert tx1.amount == Decimal("1000")
            assert tx1.activity_type == ActivityType.TRANSACTION

        await collector.close()

    @pytest.mark.asyncio
    async def test_fetch_transactions_empty_result(
        self, etherscan_config, retry_config, mock_empty_response
    ):
        """Handle empty transaction results gracefully."""
        collector = EtherscanCollector(etherscan_config, retry_config)

        with patch.object(
            collector, "_make_request", return_value=mock_empty_response
        ):
            transactions = await collector.fetch_stablecoin_transactions(
                stablecoin="USDC",
                contract_address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                limit=100,
            )

            assert len(transactions) == 0

        await collector.close()

    @pytest.mark.asyncio
    async def test_fetch_holders_success(
        self, etherscan_config, retry_config, mock_holder_response
    ):
        """Successfully fetch and parse holders."""
        collector = EtherscanCollector(etherscan_config, retry_config)

        with patch.object(
            collector, "_make_request", return_value=mock_holder_response
        ):
            holders = await collector.fetch_token_holders(
                stablecoin="USDC",
                contract_address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                limit=100,
            )

            assert len(holders) == 2
            assert all(isinstance(h, Holder) for h in holders)

            # Check first holder
            h1 = holders[0]
            assert h1.address == "0xaaaa111111111111111111111111111111111111"
            assert h1.stablecoin == "USDC"
            assert h1.chain == "ethereum"
            assert h1.balance == Decimal("50000")

        await collector.close()

    @pytest.mark.asyncio
    async def test_fetch_transactions_api_failure(
        self, etherscan_config, retry_config
    ):
        """Handle API failure gracefully."""
        collector = EtherscanCollector(etherscan_config, retry_config)

        with patch.object(collector, "_make_request", return_value=None):
            transactions = await collector.fetch_stablecoin_transactions(
                stablecoin="USDC",
                contract_address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                limit=100,
            )

            assert len(transactions) == 0

        await collector.close()

    @pytest.mark.asyncio
    async def test_collect_all_multiple_stablecoins(
        self, etherscan_config, retry_config, mock_transaction_response
    ):
        """Collect data for multiple stablecoins."""
        collector = EtherscanCollector(etherscan_config, retry_config)

        # Mock both transaction and holder responses
        with patch.object(
            collector, "_make_request", return_value=mock_transaction_response
        ):
            stablecoins = {
                "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            }

            result = await collector.collect_all(
                stablecoins=stablecoins,
                max_records=100,
            )

            assert result.explorer_name == "etherscan"
            assert result.chain == "ethereum"
            # 2 transactions per stablecoin * 2 stablecoins = 4
            assert len(result.transactions) == 4

        await collector.close()


class TestBscscanCollector:
    """Integration tests for BscscanCollector."""

    @pytest.mark.asyncio
    async def test_fetch_transactions_success(
        self, bscscan_config, retry_config, mock_transaction_response
    ):
        """Successfully fetch and parse BSC transactions."""
        collector = BscscanCollector(bscscan_config, retry_config)

        with patch.object(
            collector, "_make_request", return_value=mock_transaction_response
        ):
            transactions = await collector.fetch_stablecoin_transactions(
                stablecoin="USDC",
                contract_address="0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
                limit=100,
            )

            assert len(transactions) == 2
            assert all(tx.chain == "bsc" for tx in transactions)
            assert all(tx.source_explorer == "bscscan" for tx in transactions)

        await collector.close()


class TestPolygonscanCollector:
    """Integration tests for PolygonscanCollector."""

    @pytest.mark.asyncio
    async def test_fetch_transactions_success(
        self, polygonscan_config, retry_config, mock_transaction_response
    ):
        """Successfully fetch and parse Polygon transactions."""
        collector = PolygonscanCollector(polygonscan_config, retry_config)

        with patch.object(
            collector, "_make_request", return_value=mock_transaction_response
        ):
            transactions = await collector.fetch_stablecoin_transactions(
                stablecoin="USDC",
                contract_address="0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
                limit=100,
            )

            assert len(transactions) == 2
            assert all(tx.chain == "polygon" for tx in transactions)
            assert all(tx.source_explorer == "polygonscan" for tx in transactions)

        await collector.close()


class TestCollectorErrorHandling:
    """Tests for collector error handling."""

    @pytest.mark.asyncio
    async def test_rate_limit_handling(
        self, etherscan_config, retry_config, mock_rate_limit_response
    ):
        """Collector handles rate limiting correctly."""
        collector = EtherscanCollector(etherscan_config, retry_config)

        # First call returns rate limit, second returns success
        call_count = 0
        success_response = {
            "status": "1",
            "message": "OK",
            "result": [],
        }

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_rate_limit_response
            return success_response

        with patch.object(collector, "_make_request", side_effect=mock_request):
            # The collector should handle rate limiting internally
            transactions = await collector.fetch_stablecoin_transactions(
                stablecoin="USDC",
                contract_address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                limit=100,
            )

            # Should return empty list (from success response)
            assert isinstance(transactions, list)

        await collector.close()

    @pytest.mark.asyncio
    async def test_invalid_transaction_data_skipped(
        self, etherscan_config, retry_config
    ):
        """Invalid transaction data is skipped gracefully."""
        collector = EtherscanCollector(etherscan_config, retry_config)

        response_with_invalid = {
            "status": "1",
            "message": "OK",
            "result": [
                {
                    "hash": "0xvalid123",
                    "blockNumber": "18500000",
                    "timeStamp": "1699000000",
                    "from": "0x1111111111111111111111111111111111111111",
                    "to": "0x2222222222222222222222222222222222222222",
                    "value": "1000000000",
                },
                {
                    # Missing required fields
                    "hash": "",
                    "blockNumber": "invalid",
                },
            ],
        }

        with patch.object(
            collector, "_make_request", return_value=response_with_invalid
        ):
            transactions = await collector.fetch_stablecoin_transactions(
                stablecoin="USDC",
                contract_address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                limit=100,
            )

            # Only valid transaction should be returned
            assert len(transactions) == 1
            assert transactions[0].transaction_hash == "0xvalid123"

        await collector.close()


class TestCollectorDataFlow:
    """Tests for complete data flow from collection to export."""

    @pytest.mark.asyncio
    async def test_full_collection_pipeline(
        self, etherscan_config, retry_config, mock_transaction_response
    ):
        """Test complete data collection pipeline."""
        from collectors.aggregator import DataAggregator
        from collectors.classifier import ActivityClassifier
        from collectors.models import ExplorerData

        collector = EtherscanCollector(etherscan_config, retry_config)
        aggregator = DataAggregator()
        classifier = ActivityClassifier()

        with patch.object(
            collector, "_make_request", return_value=mock_transaction_response
        ):
            # Step 1: Collect data
            explorer_data = await collector.collect_all(
                stablecoins={"USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"},
                max_records=100,
            )

            assert isinstance(explorer_data, ExplorerData)
            assert len(explorer_data.transactions) > 0

            # Step 2: Classify transactions
            for tx in explorer_data.transactions:
                classified_type = classifier.classify_transaction(tx)
                assert classified_type in ActivityType

            # Step 3: Aggregate data
            aggregated = aggregator.aggregate([explorer_data])

            assert aggregated.total_records > 0
            assert "etherscan" in aggregated.explorers_queried
            assert "USDC" in aggregated.by_stablecoin

            # Step 4: Verify JSON serialization
            data_dict = aggregated.to_dict()
            assert "summary" in data_dict
            assert "transactions" in data_dict

        await collector.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
