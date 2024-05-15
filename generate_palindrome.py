import random
import string
import pandas as pd

def generate_palindrome(length):
    half = ''.join(random.choices(string.ascii_lowercase, k=length//2))
    return half + half[::-1]

def generate_non_palindrome(length):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

# Generate data
data = []
for _ in range(5000):  # Adjust the number based on needs
    length = random.randint(3, 10)  # Length of the palindrome
    if random.random() > 0.5:
        data.append((generate_palindrome(length), 1))
    else:
        data.append((generate_non_palindrome(length), 0))


df = pd.DataFrame(data, columns=['text', 'label'])
df.to_csv('palindrome_dataset.csv', index=False)
