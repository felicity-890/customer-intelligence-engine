import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset

df = pd.read_csv("data/online_shoppers_intention.csv")

# Calculate average product-page duration by purchase outcome

avg_duration = df.groupby("Revenue")["ProductRelated_Duration"].mean()

# Make the chart

avg_duration.index = ["No Purchase", "Purchase"]
avg_duration.plot(kind="bar")

plt.title("Average Product-Page Duration by Purchase Outcome")
plt.xlabel("Session Outcome")
plt.ylabel("Average Duration")
plt.xticks(rotation=0)
plt.tight_layout()

# Save the chart

plt.savefig("product_duration_by_revenue.png", dpi=300)
plt.show()
