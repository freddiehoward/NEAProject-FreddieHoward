import pytest
import numpy as np
from batchingData import create_batches

# Sample data for testing
X = np.random.rand(100, 4)  # 100 samples with 4 features each
y = np.random.rand(100)     # 100 output values
batch_size = 20             # Desired batch size

def test_batch_creation():
    '''
    Test that create_batches splits the data correctly:
    - Check the number of batches.
    - Validate input-output alignment within each batch.
    '''
    batches = create_batches(X, y, batch_size)
    assert len(batches) == len(X) // batch_size, "Number of batches should be floor(X.size / batch_size)"
    for X_batch, y_batch in batches:
        assert len(X_batch) == batch_size, "Each batch should have the defined batch size"
        assert len(X_batch) == len(y_batch), "Input and output batch sizes should match"
    print("Batch creation test passed!")

def test_incomplete_last_batch():
    '''
    Test that incomplete batches are excluded.
    '''
    batches = create_batches(X[:-5], y[:-5], batch_size)
    assert len(batches[-1][0]) == batch_size, "Only complete batches should be included"
    print("Incomplete last batch exclusion test passed!")