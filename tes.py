# test_csv_processor.py

import pandas as pd
import io
from app import CSVProcessor

def test_csv_processing():
    # Simulated uploaded file
    csv_content = """name,age,salary,join_date
Alice,30,70000,2020-01-15
Bob,45,90000,2018-06-23
Charlie,28,65000,2021-09-10
"""
    fake_file = io.StringIO(csv_content)

    processor = CSVProcessor()
    df = processor.load_and_process_csv(fake_file)

    assert df is not None, "DataFrame should not be None"
    assert df.shape == (3, 4), f"Expected shape (3, 4), got {df.shape}"
    assert "salary" in processor.metadata['numeric_columns']
    assert "join_date" in processor.metadata['datetime_columns']
    assert processor.metadata['summary_stats']['salary']['mean'] == 75000.0

    print("All assertions passed.")
    print("\nMetadata:")
    for k, v in processor.metadata.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    test_csv_processing()
