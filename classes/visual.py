import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.stats import zscore
import streamlit as st


#import plotly.express as px

from utils.sentences import format_metric
from classes.data_point import Player, Country, Person
from classes.data_source import PlayerStats, CountryStats, PersonStat
from typing import Union


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

def rgb_to_color(rgb_color: tuple, opacity=1):
    return f"rgba{(*rgb_color, opacity)}"

def wrap_text(text, max_len=15):
    words = text.split()
    wrapped_text = ""
    current_len = 0
    for word in words:
        if current_len + len(word) > max_len:
            wrapped_text += "<br>"
            current_len = 0
        wrapped_text += word + " "
        current_len += len(word) + 1
    return wrapped_text.strip()

class DistributionPlot:
    def __init__(self, dataframe, entity,  metrics,explanation_provider=None,labels=None, selected_entity=None, *args, **kwargs):
        self.cols = metrics
        self.dataframe = dataframe.df
        self.entity = entity.ser_metrics
        self.background = hex_to_rgb("#faf9ed")
        self.color = hex_to_rgb("#eddefa")
        self.explanation_provider = explanation_provider
        self.labels= labels
        self.selected_entity = selected_entity
        self.fig = go.Figure()
        self.set_visualization()
        super().__init__(*args, **kwargs)
        self._setup_axes()

    def show(self):
        st.plotly_chart(
            self.fig,
            config={"displayModeBar": False},
            use_container_width=True,
        )

    def _setup_axes(self):
        if self.labels is  None:
            labels=["Worse", "Average", "Better"]
            self.fig.update_xaxes(
            range=[-3, 4],
            fixedrange=True,
            tickmode='array',
            tickvals=[-3, 0, 3],
            ticktext=labels,
            tickfont=dict(size=9),
            showgrid=False,
            gridcolor=rgb_to_color(hex_to_rgb("#6a5acd"), 0.7),
            )
        else:
            self.fig.update_xaxes(
            range=[-5, 5],
            fixedrange=True,
            showgrid=False,
            gridcolor=rgb_to_color(hex_to_rgb("#6a5acd"), 0.7),
            )
        if self.labels is not None:
            # Hide metric names if we have custom left/right labels
            self.fig.update_yaxes(
                showticklabels=False,
                fixedrange=True,
                showgrid=False,
                gridcolor=rgb_to_color(self.background),
                zerolinecolor=rgb_to_color(hex_to_rgb("#ffffff")),
            )
        else:
            # Keep default tick labels
            self.fig.update_yaxes(
                showticklabels=True,
                fixedrange=True,
                showgrid=False,
                gridcolor=rgb_to_color(self.background),
                zerolinecolor=rgb_to_color(hex_to_rgb("#ffffff")),
            )

    def get_explanation_text(self, metric_name, value):
        """Safely get explanation text or return empty string."""
        if self.explanation_provider is None:
            return ""
        try:
            return self.explanation_provider.get_explanation(metric_name, self.entity, value)
        except Exception as e:
            print(f"Explanation error for {metric_name}: {e}")
            return ""



    def set_visualization(self):
        dataframe = self.dataframe.iloc[:, -2*len(self.cols):-len(self.cols)]
        df_entity = self.entity.iloc[-2*len(self.cols):-len(self.cols)]
        df_entity_rank = self.entity.iloc[-len(self.cols):]

        # Color palette
        colors = px.colors.qualitative.Dark2 #px.colors.qualitative.Set2

        # Create subplots
        self.fig = make_subplots(
            rows=len(self.cols),
            cols=1,
            shared_xaxes=True,  # Keep the same scale for all
            vertical_spacing=0.02,
            row_heights=[1/len(self.cols)]*len(self.cols)
        )
        
        for i, col in enumerate(dataframe.columns):
            
            self.fig.add_trace(
                go.Violin(
                    x=dataframe[col].tolist(),
                    name = self.cols[i],
                    #name=wrap_text(self.cols[i]),
                    marker=dict(color=colors[i % len(colors)]),
                    opacity=0.65,
                    side='positive',
                    showlegend = False,
                    # hovertemplate=f"<b>{self.cols[i]}</b><br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>"
                    hoverinfo="skip", hovertemplate=None,
                    # hoveron="all",
                    points=False,
                ),
                row=i+1,
                col=1
            )

            # Entity marker
            entity_value = float(df_entity.iloc[i])
            rank_value=round(float(df_entity_rank.iloc[i]))
            

            explanation=self.get_explanation_text(col,entity_value)
            hovertext=(
                  f"<b>{self.cols[i].capitalize().replace('_z', ' ')}</b><br>Value: {entity_value:.3f}<br>Rank: {rank_value}"
                + (f"<br><i>{explanation}</i>" if explanation else "")
                + "<extra></extra>"
            )
            self.fig.add_trace(
                go.Scatter(
                    x=[entity_value],
                    y=[self.cols[i]],
                    mode="markers", # if we want marker and text do "markers+text"
                    marker=dict(symbol="diamond", size=9, color="#9340ff"),
                    name=self.selected_entity,
                    showlegend=(i == 0),  # ensures legend is shown
                    hovertemplate=hovertext,
                    customdata=[round(float(df_entity_rank.iloc[i]))]
                    ),
                    row=i+1,
                    col=1
                    )
            # self.fig.add_annotation(
            #     text=explanation,
            #     x=1.5, y=0.75,   # relative placement inside subplot
            #     xref=f"x{i+2} domain",
            #     yref=f"y{i+2} domain",
            #     showarrow=False,
            #     font=dict(size=12, color="black"),
            #     align="center",
            #     bgcolor="rgba(237,222,250,0.3)",  # same color palette you used
            #     bordercolor="#9340ff",
            #     borderwidth=1,
            #     )


            # --- Add explanation text to the right side of each violin, if it exists ---
            if explanation.replace("<br>", "").strip():  # only if there's text
                self.fig.add_annotation(
                    text=explanation,
                    x=1.2,  # slightly outside the plot area
                    y=self.cols[i],
                    xref="paper",  # anchor to figure, not subplot
                    yref=f"y{i+1}",  # reference the subplot’s y-axis
                    showarrow=False,
                    font=dict(size=10, color=colors[i % len(colors)]),
                    align="left",
                    xanchor="left",
                    yanchor="middle",
                    bgcolor="rgba(255,255,255,0.7)",  # subtle background for readability
                    bordercolor=colors[i % len(colors)],
                    borderwidth=1,
                    borderpad=4
                )
                


             # Add left/right labels --
            if self.labels is not None:
                left_label, right_label = self.labels.get(self.cols[i], ("", ""))



                if left_label or right_label:
                    # Left-side label
                    
                    self.fig.add_annotation(
                        text=left_label,
                        xref=f"x{i+1}",
                        yref=f"y{i+1}",
                        x=-4.01,  # set annotation x position
                        y=self.cols[i],
                        showarrow=False,
                        font=dict(size=11, color="gray"),
                        xanchor="right",
                        yshift=-10  # Slightly lower position
                    )

                    # Right-side label
                    self.fig.add_annotation(
                        text=right_label,
                        xref=f"x{i+1}",
                        yref=f"y{i+1}",
                        x=4.01,  
                        y=self.cols[i],
                        showarrow=False,
                        font=dict(size=11, color="gray"),
                        xanchor="left",
                        yshift=-10  # Slightly lower position
                    )         
                    # Center label
                    self.fig.add_annotation(
                        text=f"{self.cols[i].capitalize().replace('_z', '')}",
                        x=0,  # centered horizontally
                        y=0.25,
                        xref=f"x{i+1}",
                        yref=f"y{i+1}",
                        showarrow=False,
                        font=dict(size=11, color="black"),
                        xanchor="center",
                        yanchor="bottom"
                    )


            
        # Update layout
        self.fig.update_layout(
            template="plotly_white",
            showlegend=True,
            font = dict(size=14),
            # autosize=True,
            legend=dict(
                yanchor="bottom",
                y=-0.2,
                xanchor="right",
                x=1,
                font=dict(size=14)
            ),
            margin=dict(t=50, b=50, l=45, r=355),
            height = 500,
        )

    
        # Add grid & font styling
        self.fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
        # self.fig.update_yaxes(showgrid=False)

        if self.labels is not None:
            # Only hide metric names if we have left/right labels
            self.fig.update_yaxes(showgrid=False, showticklabels=False)
        else:
            # Keep metric names visible
            self.fig.update_yaxes(showgrid=False , showticklabels=True)




class RadarPlot:
    def __init__(self, entity, metrics, explanation_provider=None, *args, **kwargs):

        self.cols = metrics
        self.entity = entity.ser_metrics
        self.explanation_provider = explanation_provider
        self.color = hex_to_rgb("#faf9ed")
        self.fig = go.Figure()
        super().__init__(*args, **kwargs)
        self.set_visualization()
     

    def show(self):
        st.plotly_chart(
            self.fig,
            config={"displayModeBar": False},
            use_container_width=True,
        )

    # def get_explanation_text(self, metric_name, value):
    #     """Safely get explanation text or return empty string."""
    #     if self.explanation_provider is None:
    #         return ""
    #     try:
    #         return self.explanation_provider.get_explanation(metric_name, self.entity, value)
    #     except Exception as e:
    #         print(f"Explanation error for {metric_name}: {e}")
    #         return ""


    def set_visualization(self):
        # Streamlit primary color
        color = st.get_option("theme.primaryColor")
        if color is None:
            color = "#FF4B4B"

    
        df_entity = self.entity.iloc[-2*len(self.cols):-len(self.cols)]
        r_values = df_entity.values.tolist()
        #theta_values=self.cols
        theta_values=[wrap_text(c) for c in self.cols]

        # Repeat the first element at the end to close the polygon
        r_values.append(r_values[0])
        theta_values = theta_values + [theta_values[0]]

        hover_texts = []
        for metric, r in zip(self.cols, r_values[:-1]):  # skip closing point
            # Use explainer if available
            if self.explanation_provider:
                try:
                    explanation = self.explanation_provider.get_explanation(metric,self.entity, r)
                    hover_texts.append(f"<br>{explanation}")
                except Exception:
                    # Fallback in case explainer fails or metric missing
                    hover_texts.append(f"")
            else:
                # No explainer available
                hover_texts.append(f"")

        # Repeat hover text for closing point
        hover_texts.append(hover_texts[0])
        # Add the entity as a highlighted polygon
        self.fig.add_trace(
            go.Scatterpolar(
            r = r_values,
            theta = theta_values,
            mode="lines+markers",
            line=dict(color=rgb_to_color(hex_to_rgb("#9340ff")), width=3),
            marker=dict(size=8, color=rgb_to_color(hex_to_rgb("#9340ff"))),
            fill="toself",
            hovertext=[f"<b>{metric.capitalize()}</b>: {value:.3f}<br><i>{explanation}</i>" for metric, value, explanation in zip(self.cols + [self.cols[0]], r_values, hover_texts)],
            hoverinfo="text",
            showlegend=False,
            )
        )
        # add annotations for each metric
        # Place annotation boxes just outside the radar circle
        radius = 0.4  # slightly outside the circle (circle radius is 0.4)
        num_metrics = len(self.cols)
        for i, (theta, r, hover) in enumerate(zip(theta_values[:-1], r_values[:-1], hover_texts[:-1])):
            angle =  1* np.pi * i / num_metrics
            x = 0.5 + radius * np.cos(angle)
            y = 0.5+ radius * np.sin(angle)
            if len(hover.replace("<br>", "").strip()) > 0:
                self.fig.add_annotation(
                xref="paper",
                yref="paper",
                x=x,
                y=y,
                text=hover,
                showarrow=False,
                font=dict(size=12, color="#9340ff"),
                align="center",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#9340ff",
                borderwidth=1,
                borderpad=4,
                )


        self.fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-4, 4],
                    gridcolor=rgb_to_color(hex_to_rgb("#E0E0E0")),
                    linecolor=rgb_to_color(hex_to_rgb("#CCCCCC")),
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=10, family="Gilroy-Light", color="#333")
                )
            ),
            margin=dict(l=75, r=85, t=55, b=55),
            plot_bgcolor="white",
            showlegend=False,
            autosize=True,
        )




