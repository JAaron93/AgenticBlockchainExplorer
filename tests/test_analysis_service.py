import pytest
import pandas as pd
from decimal import Decimal
from datetime import datetime, timezone
from core.analysis_service import AnalysisService
from collectors.models import Holder

def test_classify_holders_with_duplicates_raises_error():
    service = AnalysisService()
    
    # Setup holders
    holders = [
        Holder(
            address="0x123",
            balance=Decimal("100"),
            stablecoin="USDC",
            chain="ethereum",
            first_seen=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
            is_store_of_value=False,
            source_explorer="etherscan"
        )
    ]
    
    # Setup duplicate predictions
    sov_predictions = pd.DataFrame({
        "address": ["0x123", "0x456", "0x123"],
        "prediction": [1, 0, 0]
    })
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="Duplicate addresses found in sov_predictions"):
        service.classify_holders(
            holders=holders,
            transactions=[],
            use_ml_results=True,
            sov_predictions=sov_predictions
        )

def test_classify_holders_normal_operation():
    service = AnalysisService()
    
    # Setup holders
    holders = [
        Holder(
            address="0x123",
            balance=Decimal("100"),
            stablecoin="USDC",
            chain="ethereum",
            first_seen=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
            is_store_of_value=False,
            source_explorer="etherscan"
        )
    ]
    
    # Setup valid predictions
    sov_predictions = pd.DataFrame({
        "address": ["0x123", "0x456"],
        "prediction": [1, 0]
    })
    
    # Should work normally
    result = service.classify_holders(
        holders=holders,
        transactions=[],
        use_ml_results=True,
        sov_predictions=sov_predictions
    )
    
    assert result[0].is_store_of_value is True

if __name__ == "__main__":
    pytest.main([__file__])
