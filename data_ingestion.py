import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import time

# PostgreSQL Connection String
DATABASE_URI = 'sqlite:///churn_db.sqlite'
ADMIN_EMAIL = 'phusevinayak2004@gmail.com'

def generate_synthetic_data(num_records=5000):
    np.random.seed(42)
    
    data = {
        'customer_id': range(1, num_records + 1),
        'age': np.random.randint(18, 70, size=num_records),
        'tenure_months': np.random.randint(1, 60, size=num_records),
        'monthly_charges': np.random.uniform(20.0, 120.0, size=num_records),
        'total_charges': np.zeros(num_records),
        'support_tickets': np.random.randint(0, 10, size=num_records),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=num_records, p=[0.5, 0.3, 0.2]),
        'has_internet_service': np.random.choice([1, 0], size=num_records, p=[0.8, 0.2]),
        'churn': np.zeros(num_records)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate total charges (simplification)
    df['total_charges'] = df['tenure_months'] * df['monthly_charges']
    
    # Introduce some logic for churn probability based on features
    # Higher churn for Month-to-month, high support tickets, high monthly charges
    churn_prob = (
        0.3 * (df['contract_type'] == 'Month-to-month') +
        0.1 * (df['support_tickets'] > 3) +
        0.05 * (df['monthly_charges'] > 80) -
        0.1 * (df['tenure_months'] > 24)
    )
    
    # Normalize probabilities and assign churn
    churn_prob = np.clip(churn_prob, 0, 1)
    df['churn'] = np.random.binomial(1, churn_prob)
    
    # Ensure our required user exists in the system as an admin/test account
    # We can add an 'email' column or just use it as project metadata
    df['email'] = [f'user_{i}@example.com' for i in range(num_records)]
    df.loc[0, 'email'] = ADMIN_EMAIL
    
    return df

def ingest_data():
    print("Generating synthetic data...")
    df = generate_synthetic_data(5000)
    
    print("Connecting to database...")
    engine = create_engine(DATABASE_URI)
    
    # Wait for DB to be ready
    max_retries = 5
    for i in range(max_retries):
        try:
            with engine.connect() as conn:
                print("Connection successful!")
                break
        except Exception as e:
            if i == max_retries - 1:
                raise e
            print(f"Database not ready. Retrying in 5 seconds... ({i+1}/{max_retries})")
            time.sleep(5)
            
    print("Writing data to 'customers' table...")
    df.to_sql('customers', engine, if_exists='replace', index=False)
    print("Data ingestion complete!")

if __name__ == '__main__':
    ingest_data()
