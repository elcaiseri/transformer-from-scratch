from datasets import load_dataset
import pandas as pd

def load_sample_data(sample_size=10000):
    dataset = load_dataset('wmt14', 'de-en', split='test')

    data = []
    for i, example in enumerate(dataset):
        if i >= sample_size:
            break
        data.append((example['translation']['en'], example['translation']['de']))

    df = pd.DataFrame(data, columns=['english', 'german'])
    return df

if __name__ == "__main__":
    sample_df = load_sample_data()
    print(f"Loaded {len(sample_df)} samples")
    print(sample_df.head())
    
    # Save the DataFrame to a CSV file
    sample_df.to_csv('/./data/sample_data.csv', index=False)
    print("DataFrame saved to data")
