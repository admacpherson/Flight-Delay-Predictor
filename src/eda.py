import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROCESSED_DIR = "data/processed"
FILENAME = "cleaned_T_ONTIME_MARKETING.csv"

EXPORT_DIR = "figures/"

df = pd.read_csv(os.path.join(PROCESSED_DIR, FILENAME))

# Plot distribution of arrival delay
sns.histplot(df["ARR_DELAY"], bins=50, kde=True)
plt.title("Distribution of Arrival Delays")
plt.xlabel("Delay (minutes)")
plt.ylabel("Count")
plt.axvline(x=15, color='red', linestyle='--', label='15 min cutoff')
plt.legend()
plt.tight_layout()
plt.savefig(EXPORT_DIR + "Delay_Histogram")
plt.show()

# Plot delays by airline
plt.figure(figsize=(10, 5))
sns.boxplot(x="OP_UNIQUE_CARRIER", y="ARR_DELAY", data=df)
plt.title("Arrival Delay by Airline")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(EXPORT_DIR + "Delays_by_Airline")
plt.show()