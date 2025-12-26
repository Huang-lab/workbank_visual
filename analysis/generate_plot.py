import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# Calculate a "Priority Score"
merged_df['Priority Score'] = merged_df['Automation Desire Rating'] * merged_df['Automation Capacity Rating']
merged_df = merged_df.sort_values('Priority Score', ascending=False)

# Create the figure
fig = go.Figure()

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
            textposition="top center",
            customdata=np.stack((occ_data['Priority Score'],), axis=-1),
            hovertemplate="<b>%{text}</b><br>Occupation: " + occ + "<br>Capability: %{x:.2f}<br>Desire: %{y:.2f}<br>Priority: %{customdata[0]:.2f}<extra></extra>",
            marker=dict(size=12, opacity=0.8, color=colors[i % len(colors)]),
            legendgroup=occ,
        )
    )

# Create Dropdown Buttons
buttons = []

# "All Occupations" Button
buttons.append(dict(
    method="update",
    label="All Occupations",
    args=[
        {"visible": [True] * len(occupations), "mode": ["markers"] * len(occupations)},
        {"title": "WORKBank: All Occupations"}
    ]
))

# Individual Occupation Buttons
for i, occ in enumerate(occupations):
    occ_data = merged_df[merged_df['Occupation (O*NET-SOC Title)'] == occ]
    num_tasks = len(occ_data)
    
    # Visibility: only this occupation is True
    visible = [o == occ for o in occupations]
    
    # Mode: markers+text if < 10 tasks, else markers
    modes = []
    for j, o in enumerate(occupations):
        if o == occ:
            modes.append('markers+text' if num_tasks < 10 else 'markers')
        else:
            modes.append('markers')
            
    buttons.append(dict(
        method="update",
        label=occ[:40] + "..." if len(occ) > 40 else occ,
        args=[
            {"visible": visible, "mode": modes},
            {"title": f"WORKBank: {occ} ({num_tasks} tasks)"}
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
        y=1.15,
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
    xaxis=dict(
        title="AI Expert-rated Capability", 
        range=[0.5, 5.5],
        gridcolor='rgba(0,0,0,0.05)'
    ),
    yaxis=dict(
        title="Worker-related Desire", 
        range=[0.5, 5.5],
        gridcolor='rgba(0,0,0,0.05)'
    ),
    height=900,
    width=1200,
    template="plotly_white",
    showlegend=True,
    legend=dict(
        title="Occupations",
        font=dict(size=10),
        y=0.5,
        x=1.02,
        xanchor="left"
    ),
    margin=dict(l=50, r=250, t=150, b=50)
)

# Add quadrant lines
fig.add_shape(type="line", x0=3, y0=0.5, x1=3, y1=5.5, line=dict(color="rgba(0,0,0,0.2)", dash="dash"))
fig.add_shape(type="line", x0=0.5, y0=3, x1=5.5, y1=3, line=dict(color="rgba(0,0,0,0.2)", dash="dash"))

# Save to HTML
os.makedirs("public", exist_ok=True)
output_path = os.path.abspath("public/index.html")
fig.write_html(output_path, include_plotlyjs='cdn')

print(f"Cleaned interactive plot saved to: {output_path}")
