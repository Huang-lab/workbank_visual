import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datasets import load_dataset
import os
import numpy as np

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

# Calculate a "Priority Score" (Desire * Capability) to identify "Low Hanging Fruit"
merged_df['Priority Score'] = merged_df['Automation Desire Rating'] * merged_df['Automation Capacity Rating']
merged_df = merged_df.sort_values('Priority Score', ascending=False)

# Create the figure with subplots: Scatter on left, Table on right
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.65, 0.35],
    horizontal_spacing=0.05,
    specs=[[{"type": "scatter"}, {"type": "table"}]]
)

# Get unique occupations
occupations = sorted(merged_df['Occupation (O*NET-SOC Title)'].unique())
colors = px.colors.qualitative.Alphabet

# Add Scatter traces for each occupation
for i, occ in enumerate(occupations):
    occ_data = merged_df[merged_df['Occupation (O*NET-SOC Title)'] == occ]
    fig.add_trace(
        go.Scatter(
            x=occ_data['Automation Capacity Rating'],
            y=occ_data['Automation Desire Rating'],
            mode='markers',
            name=occ,
            text=occ_data['Task'],
            customdata=np.stack((occ_data['Priority Score'],), axis=-1),
            hovertemplate="<b>%{text}</b><br>Occupation: " + occ + "<br>Capability: %{x:.2f}<br>Desire: %{y:.2f}<br>Priority: %{customdata[0]:.2f}<extra></extra>",
            marker=dict(size=10, opacity=0.8, color=colors[i % len(colors)]),
            legendgroup=occ,
        ),
        row=1, col=1
    )

# Add a Table trace (initially showing top 20 tasks overall)
top_tasks = merged_df.head(20)
fig.add_trace(
    go.Table(
        header=dict(
            values=["<b>Task (Top 20)</b>", "<b>Desire</b>", "<b>Cap.</b>", "<b>Score</b>"],
            fill_color='#2c3e50',
            align='left',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[top_tasks['Task'], 
                    top_tasks['Automation Desire Rating'].round(2), 
                    top_tasks['Automation Capacity Rating'].round(2),
                    top_tasks['Priority Score'].round(2)],
            fill_color='#f8f9fa',
            align='left',
            font=dict(size=10)
        )
    ),
    row=1, col=2
)

# Create Dropdown Buttons
buttons = []

# "All Occupations" Button
buttons.append(dict(
    method="update",
    label="All Occupations",
    args=[
        {"visible": [True] * len(occupations) + [True]}, 
        {
            "title": "WORKBank: All Occupations (Sorted by Priority)",
            "cells.values": [
                merged_df.head(20)['Task'], 
                merged_df.head(20)['Automation Desire Rating'].round(2), 
                merged_df.head(20)['Automation Capacity Rating'].round(2),
                merged_df.head(20)['Priority Score'].round(2)
            ]
        }
    ]
))

# Individual Occupation Buttons
for occ in occupations:
    visible = [o == occ for o in occupations] + [True]
    occ_top = merged_df[merged_df['Occupation (O*NET-SOC Title)'] == occ].head(20)
    
    buttons.append(dict(
        method="update",
        label=occ[:40] + "..." if len(occ) > 40 else occ,
        args=[
            {"visible": visible},
            {
                "title": f"WORKBank: {occ}",
                "cells.values": [
                    occ_top['Task'], 
                    occ_top['Automation Desire Rating'].round(2), 
                    occ_top['Automation Capacity Rating'].round(2),
                    occ_top['Priority Score'].round(2)
                ]
            }
        ]
    ))

# Update Layout
fig.update_layout(
    updatemenus=[dict(
        buttons=buttons,
        direction="down",
        showactive=True,
        x=0.0,
        xanchor="left",
        y=1.1,
        yanchor="top",
        bgcolor="white",
        bordercolor="gray"
    )],
    title=dict(
        text="<b>WORKBank: Automation Landscape</b>",
        font=dict(size=24),
        x=0.5,
        y=0.98
    ),
    xaxis=dict(title="AI Expert-rated Capability", range=[0.8, 5.2]),
    yaxis=dict(title="Worker-related Desire", range=[0.8, 5.2]),
    height=800,
    width=1400,
    template="plotly_white",
    showlegend=True,
    legend=dict(
        title="Occupations",
        font=dict(size=8),
        y=0.5,
        x=1.1,
        xanchor="left"
    ),
    margin=dict(l=50, r=350, t=100, b=50)
)

# Add quadrant lines to the scatter plot (row 1, col 1)
fig.add_shape(type="line", x0=3, y0=1, x1=3, y1=5, line=dict(color="rgba(0,0,0,0.2)", dash="dash"), row=1, col=1)
fig.add_shape(type="line", x0=1, y0=3, x1=5, y1=3, line=dict(color="rgba(0,0,0,0.2)", dash="dash"), row=1, col=1)

# Save to HTML
os.makedirs("public", exist_ok=True)
output_path = os.path.abspath("public/index.html")
fig.write_html(output_path, include_plotlyjs='cdn')

print(f"Advanced interactive plot saved to: {output_path}")
