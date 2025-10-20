from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(42)
data = np.arange(100)
print("Original data indices: 0-99\n")

# Method 1: 100 -> 60/40, then 40 -> 20/20
train1, temp1 = train_test_split(data, test_size=0.4, random_state=1)
val1, test1 = train_test_split(temp1, test_size=0.5, random_state=1)

# Method 2: 100 -> 80/20, then 80 -> 60/20
temp2, test2 = train_test_split(data, test_size=0.2, random_state=1)
train2, val2 = train_test_split(temp2, test_size=0.25, random_state=1)

print(f"Train overlap: {len(set(train1) & set(train2))}/{len(train1)}")
print(f"Val overlap: {len(set(val1) & set(val2))}/{len(val1)}")
print(f"Test overlap: {len(set(test1) & set(test2))}/{len(test1)}")
