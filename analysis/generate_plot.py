import pandas as pd
import plotly.express as px
from datasets import load_dataset
import os

# Load datasets
worker_desire_df = load_dataset("SALT-NLP/WORKBank", data_files="worker_data/domain_worker_desires.csv")["train"].to_pandas()
expert_ratings_df = load_dataset("SALT-NLP/WORKBank", data_files="expert_ratings/expert_rated_technological_capability.csv")["train"].to_pandas()

# Aggregate worker desire by Task
worker_agg = worker_desire_df.groupby("Task").agg({
    "Automation Desire Rating": "mean",
    "Occupation (O*NET-SOC Title)": "first"
}).reset_index()

# Aggregate expert capability by Task
expert_agg = expert_ratings_df.groupby("Task").agg({
    "Automation Capacity Rating": "mean"
}).reset_index()

# Merge datasets
merged_df = pd.merge(worker_agg, expert_agg, on="Task")

# Create interactive scatter plot
fig = px.scatter(
    merged_df,
    x="Automation Capacity Rating",
    y="Automation Desire Rating",
    color="Occupation (O*NET-SOC Title)",
    hover_name="Task",
    hover_data={
        "Occupation (O*NET-SOC Title)": True,
        "Automation Capacity Rating": ":.2f",
        "Automation Desire Rating": ":.2f"
    },
    labels={
        "Automation Capacity Rating": "AI Expert-rated Capability",
        "Automation Desire Rating": "Worker-related Desire",
        "Occupation (O*NET-SOC Title)": "Occupation"
    },
    title="<b>WORKBank: Automation Landscape</b><br><sup>Worker Desire vs. AI Expert Capability across 104 Occupations</sup>",
    template="plotly_white",
    opacity=0.8,
    color_discrete_sequence=px.colors.qualitative.Alphabet # Use a large palette for many occupations
)

# Add diagonal line for reference (Desire = Capability)
fig.add_shape(
    type="line", line=dict(dash="dash", color="rgba(0,0,0,0.3)", width=1.5),
    x0=1, y0=1, x1=5, y1=5,
    layer="below"
)

# Add quadrant annotations
fig.add_annotation(x=4.5, y=4.5, text="High Desire / High Capability", showarrow=False, font=dict(color="gray"))
fig.add_annotation(x=1.5, y=1.5, text="Low Desire / Low Capability", showarrow=False, font=dict(color="gray"))

# Update layout for better readability
fig.update_layout(
    xaxis=dict(range=[0.8, 5.2], gridcolor='rgba(0,0,0,0.05)', zeroline=False),
    yaxis=dict(range=[0.8, 5.2], gridcolor='rgba(0,0,0,0.05)', zeroline=False),
    width=1100,
    height=800,
    legend=dict(
        title_font_family="Arial",
        font=dict(size=10),
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02
    ),
    margin=dict(l=50, r=250, t=100, b=50),
    title_font=dict(size=24)
)

# Save to HTML in a 'public' directory for Vercel
os.makedirs("public", exist_ok=True)
output_path = os.path.abspath("public/index.html")
fig.write_html(output_path, include_plotlyjs='cdn')

print(f"Enhanced plot saved to: {output_path}")
