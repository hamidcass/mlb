import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_prep import prep_b_data
from train_models import pipeline 



st.set_page_config(page_title="MLB Performance Projections", layout="wide", page_icon="")

# Title
st.title("2025 MLB Performance Projections")
st.markdown("---")

# Horizontal configuration bar
col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 2])

with col1:
    target_stat = st.selectbox(
        "Target Stat",
        ["OPS", "HR", "AVG", "wRC+"],
        key="target_stat"
    )

with col2:
    model_choice = st.selectbox(
        "Model",
        ["XGBoost", "Random Forest", "Ridge Regression", "Linear Regression"],
        key="model"
    )
#TODO: make this affect data (it is a dud rn)

# with col3:
#     year_range = st.select_slider(
#         "Training Years",
#         options=list(range(2020, 2025)),
#         value=(2020, 2024),
#         key="years"
#     )
year_range = (2020, 2024)

with col4:
    st.write("")  # Spacing
    st.write("")  # Spacing

with col5:
    st.write("")  # Spacing
    run_button = st.button("Run Predictions", type="primary", use_container_width=True)

st.markdown("---")

# Results section
if run_button or 'results' in st.session_state:
    if run_button:
        # TODO: Run your model predictions here
        # For now, using placeholder data

        #send target stat to prep file to get new df of appropriate features
        df = prep_b_data.run(target_stat)


        
        #now that we have the df with correct features, send to pipeline
        mae, r2, num_players, results_df, importance_df = pipeline.run(df, target_stat, model_choice, year_range)
        
        st.session_state.results = {
            "training_data": df,
            "importance_df": importance_df,
            "year_range": year_range,
            'mae': mae,
            'r2': r2,
            'num_players': num_players,
            "results_df": results_df,
        }
    
    # Metrics row
    st.subheader(f"{year_range[1]+1} {target_stat} Predictions - {model_choice}")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("MAE", f"{st.session_state.results['mae']:.4f}")
    with metric_col2:
        st.metric("R²", f"{st.session_state.results['r2']:.3f}")
    with metric_col3:
        st.metric("Players", st.session_state.results['num_players'])
    
    st.markdown("")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Summary", "Player Search", "Feature Analysis"])
    
    with tab1:
        st.subheader("Predicted vs Actual Performance")
        
        # # TODO: Replace with your actual scatter plot
        # # Placeholder visualization
        # dummy_data = pd.DataFrame({
        #     'Actual': np.random.uniform(0.5, 1.1, 100),
        #     'Predicted': np.random.uniform(0.6, 1.0, 100)
        # })

        #use plot data for visualizations
        
        fig = px.scatter(
            st.session_state.results['results_df'], 
            x='Actual', 
            y='Predicted',
            hover_data={
                'Player': True,
                f'{year_range[1]+1} Team': True,
                'Actual': ':.3f',
                'Predicted': ':.3f',
                "Error": ":.3f"
    },
            title=f"Projected vs Actual {target_stat}",
            labels={'Actual': f'Actual {target_stat}', 'Projected': f'Projected {target_stat}'}
        )
        
        def get_scale():
                if target_stat == "AVG":
                    return [0.15, 0.35]
                elif target_stat == "HR":
                    return [0, 65]
                elif target_stat == "OPS":
                    return [0.4, 1.2]
                elif target_stat == "wRC+":
                    return [20, 180]

        # Add perfect prediction line
        fig.add_trace(go.Scatter(
             
            x=get_scale(),
            y=get_scale(),
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))

        #TODO: Add model trend line
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top/Bottom performers

        #apply colours
        def colour_cells(value):
            if value < 0: #underperformane
                color = "red"
            elif value > 0:
                color = "blue"
            else:
                color = ""
            return f"background-color: {color}"

        col_a, col_b = st.columns(2)
        with col_a:

            results_df = st.session_state.results['results_df']

            st.markdown("**Top 5 Overperformers**")

            overperformers = (
                results_df[results_df['Error'] < 0]
                .nsmallest(5, 'Error') #fixes sorting bug
                .head(5)[['Player', 'Predicted', 'Actual', 'Error']]
                .copy()
            )

            styled = overperformers.style.applymap(
                colour_cells, subset=["Error"]
            )
            
            #limit to appropriate decimal place for stat
            if target_stat in ['OPS', 'AVG']:
                styled = styled.format({
                    "Predicted": "{:.3f}",
                    "Actual": "{:.3f}",
                    "Error": "{:.3f}",
                })
            else:  # HR, wrc+ (whole numbers)
                styled = styled.format({
                    "Predicted": "{:.0f}",
                    "Actual": "{:.0f}",
                    "Error": "{:.0f}",
                })

            st.dataframe(styled, hide_index=True)

        with col_b:

            results_df = st.session_state.results['results_df']

            st.markdown("**Top 5 Underperformers**")

            underperformers = (
                results_df[results_df['Error'] > 0]
                .nlargest(5, 'Error') #fixes sorting bug
                .head(5)[['Player', 'Predicted', 'Actual', 'Error']]
                .copy()
            )

            styled = underperformers.style.applymap(
                colour_cells, subset=["Error"]
            )

            if target_stat in ['OPS', 'AVG', 'wRC+']:
                styled = styled.format({
                    "Predicted": "{:.3f}",
                    "Actual": "{:.3f}",
                    "Error": "{:.3f}",
                })
            else:  # HR, counting stats
                styled = styled.format({
                    "Predicted": "{:.0f}",
                    "Actual": "{:.0f}",
                    "Error": "{:.0f}",
                })

            st.dataframe(styled, hide_index=True)
            
    with tab2:

        def get_prediction_grade(error, stat_type):
            #Return letter grade for prediction quality
            abs_error = abs(error)
            
            thresholds = {
                "OPS": [(0.030, "A+"), (0.050, "A"), (0.070, "B"), (0.090, "C"), (0.120, "D")],
                "HR": [(3, "A+"), (5, "A"), (8, "B"), (12, "C"), (15, "D")],
                "AVG": [(0.020, "A+"), (0.035, "A"), (0.050, "B"), (0.070, "C"), (0.090, "D")],
                "wRC+": [(8, "A+"), (15, "A"), (25, "B"), (35, "C"), (45, "D")]
            }
            
            for threshold, grade in thresholds[stat_type]:
                if abs_error < threshold:
                    return grade
            return "F"


        results_df = st.session_state.results['results_df']

        st.subheader("Player Search")
        
        # Searchable dropdown (filters as you type)
        player_list = ['Select a player...'] + sorted(results_df['Player'].unique().tolist())
        selected_player = st.selectbox(
            "🔍 Search or select a player",
            player_list,
            index=0
        )
        
        if selected_player and selected_player != 'Select a player...':

            results_df = st.session_state.results['results_df']
            year_range = st.session_state.results['year_range']

            # Get player data
            player_row = results_df[results_df['Player'] == selected_player].iloc[0]
            print(player_row)
            st.markdown(f"### {player_row['Player']}")
            
            # Player card

            #bio
            st.markdown("#### Player Info")
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.write(f"**Team:** {player_row[f'{year_range[1]} Team']} → {player_row[f'{year_range[1]+1} Team']}")
            with info_col2:
                st.write(f"**Age:** {player_row['Age']}")
            with info_col3:
                st.write(f"**PA:** {player_row['PA']}")
            
            st.markdown("---")

            # Prediction metrics
            st.markdown(f"#### Prediction Summary ({model_choice})")
            card_col1, card_col2, card_col3 = st.columns([1, 1, 1])

            if target_stat in ['OPS', 'AVG']:
                fmt = "{:.3f}"
            else:  # HR, wRC+
                fmt = "{:.0f}"
            
            with card_col1:
                st.metric(
                    f"Predicted {year_range[1]+1} {target_stat}", 
                    fmt.format(player_row["Predicted"])
                )
            with card_col2:
                st.metric(
                    f"Actual {year_range[1]+1} {target_stat}", 
                    fmt.format(player_row["Actual"]), 
                    delta=fmt.format(-player_row["Error"])
                )
            with card_col3:

                #getter letter grade score for guess
                score = get_prediction_grade(player_row["Error"], target_stat)
                st.metric("Prediction Score", score)


                # # Calculate confidence based on error
                # confidence = max(0, 100 - abs(player_row['Pct_Error']))
                # st.metric("Accuracy", f"{confidence:.0f}%")
            
            
            st.markdown("---")
            st.markdown("#### Historical Performance")

            df = prep_b_data.run(target_stat)
            player_history = df[df['Name'] == selected_player][['Current_Season', 'Current_Team', f"Current_{target_stat}"]].sort_values('Current_Season')
            player_history = player_history[player_history['Current_Season'].between(year_range[0], year_range[1])]

            # Add prediction to graph
            prediction_row = pd.DataFrame({
                'Current_Season': [year_range[1] + 1],
                'Current_Team': [player_row[f'{year_range[1]+1} Team']],
                f'Current_{target_stat}': [player_row['Predicted']]
            })

            # Combine historical data with prediction
            player_history_full = pd.concat([player_history, prediction_row], ignore_index=True)

            # Line graph showing historical performance + prediction
            fig = px.line(
                player_history_full, 
                x='Current_Season', 
                y=f'Current_{target_stat}',
                markers=True,
                labels={
                    'Current_Season': 'Season',
                    f'Current_{target_stat}': target_stat
                }
            )

            # Customize appearance
            fig.update_traces(
                line=dict(color='#003087', width=3),
                marker=dict(size=10, color='#13aa52'),
                name='Historical'
            )

            # Get last historical year value for connecting lines
            last_year = player_history.iloc[-1]['Current_Season']
            last_value = player_history.iloc[-1][f'Current_{target_stat}']

            # Add dashed line from 2024 to 2025 Actual
            fig.add_trace(go.Scatter(
                x=[last_year, year_range[1] + 1],
                y=[last_value, player_row['Actual']],
                mode='lines+markers',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=10, color='red'),
                name='Actual',
                hovertemplate='<b>Season:</b> %{x}<br>' +
                            f'<b>{target_stat} (Actual):</b> %{{y:.3f}}<br>' +
                            '<extra></extra>'
            ))

            # Highlight the 2025 prediction point with star
            fig.add_trace(go.Scatter(
                x=[year_range[1] + 1],
                y=[player_row['Predicted']],
                mode='markers',
                marker=dict(size=10, color='orange', symbol='circle'),
                name='Prediction',
                hovertemplate=f'<b>Season:</b> {year_range[1] + 1}<br>' +
                            f'<b>{target_stat} (Predicted):</b> {player_row["Predicted"]:.3f}<br>' +
                            '<extra></extra>'
            ))

            # Add hover info for historical line
            fig.update_traces(
                hovertemplate='<b>Season:</b> %{x}<br>' +
                            f'<b>{target_stat}:</b> %{{y:.3f}}<br>' +
                            '<extra></extra>',
                selector=dict(name='Historical')
            )

            st.plotly_chart(fig, use_container_width=True)

                        
                
        else:
            st.info("Start typing or click to select a player")
    
    with tab3:

        if model_choice in ['Linear Regression', 'Ridge Regression']:
            measurement = "Coefficient"

            st.subheader(f"Feature {measurement} Analysis")

            feature_data = st.session_state.results['importance_df']
            
            try:
                feature_data['Color'] = feature_data[measurement].apply(lambda x: 'Positive' if x > 0 else 'Negative')
                fig_features = px.bar(
                    feature_data,
                    x=measurement,
                    y="Feature",
                    orientation="h",
                    color="Color",
                    color_discrete_map={'Positive': 'green', 'Negative': 'red'},
                    title=f"Top Features for {target_stat} Prediction ({model_choice})"
                )

                #sort by value not alphabetically
               
                fig_features.update_yaxes(categoryorder='total ascending') 
                
                st.plotly_chart(fig_features, use_container_width=True)
            except Exception as e:
                st.info(f"Press **Run Predictions** above to generate feature {measurement} data.")



        else:  # tree-based models (SHAP)
            measurement = "SHAP"
            feature_data = st.session_state.results['importance_df']
            # Use signed SHAP values for directionality
            
            
            st.subheader(f"Feature {measurement} Analysis")

            try:
                feature_data['Color'] = feature_data['Direction'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
                fig_features = px.bar(
                    feature_data.sort_values('Importance', ascending=False),
                    x='Direction',      # signed SHAP value
                    y="Feature",
                    orientation="h",
                    color='Color',
                    color_discrete_map={'Positive': 'green', 'Negative': 'red'},
                    title=f"Top Features for {target_stat} Prediction ({model_choice})"
                )
                st.plotly_chart(fig_features, use_container_width=True)
            except Exception as e:
                st.info(f"Press **Run Predictions** above to generate feature {measurement} data.")
        
        
        # Conclusions about what model learned
        st.markdown("### Key Insights")
        # st.markdown(f"#### {measurement} Interpretation")
        
        if target_stat == "OPS":
            st.markdown(
                """
                - ***Agreements:*** Between all models, strikeout rate (K%) and age were among top negative influencers on OPS performance.
                - ***Disagreements:*** Tree based models heavily penalized HardHit% and xwOBA, while linear models favored them as positive contributors.
                - ***Conclusion:*** Linear models seem to more accurately understand baseball logic. Tree based models struggle more because their non-linear nature uses certain stats as anchors too heavily, leading to counterintuitive results.
                """
            )
        elif target_stat == "AVG":
            st.markdown(
                """
                - ***Agreements:*** Overall, BABIP and HardHit% showed the most positive influence between models, while age and K% were negative.
                - ***Disagreements:*** Tree based models punished PA (Plate Appearances) while linear models did the opposite.
                - ***Conclusion:*** Linear models seem to be superior for AVG prediction, as they emphasize quality contact more than tree-based models. XGBoost and Random Forest may be overfitting due to noise or failinig to understand the luck factor. These findings seem to suggest that the underlying mechanics of batting average are primarily additive and linear.
                """
            )
        elif target_stat == "HR":
            st.markdown(
                """
                - ***Agreements:*** Top positive metrics across all models included Barrel%, HardHit%, PA, Pull% and FB%. This makes sense as home runs tend to increase with more chances (PA), and better quality of contact (Barrel%, HardHit%) especially when combined with launch angle metrics (FB%, Pull%).
                - ***Disagreements:*** The linear models see age as a top-three negative factor, while Random Forest interprets it as a slight positive impact on HRs, perhaps capturing the "old man strength" or veteran power hitter profile. Additionally, tree models highlighted HR/FB as a significant positive factor, while linear models did not rank it as highly.
                - ***Conclusion:*** Random Forest had the lowest MAE and highest $R^2$, indicating that "HR power" is a result of complex interactions (you need both high EV and high FB% to see results) that tree-based models capture better than simple addition. However, XGBoost performed the worst, potentially due to being more sensitive to hyperparameters.
                """
            )
        elif target_stat == "wRC+":
            st.markdown(
                """
                - ***Agreements:*** wOBA and ISO (Isolated Power) tended to be among the top positive influencers, while age was a negative influencer across the models.
                - ***Disagreements:*** XGBoost seemed to reverse expected baseball logic, penalizing HardHit%, Barrel%, and BB% (Walk Rate), while suprisingly deeming a high K% (Strikeout Rate) as a positive contributor!
                - ***Conclusion:*** While Random Forest achieved a slightly higher $R^2$, its reliance on a single feature (wOBA) makes it less descriptive than the linear models. The Linear/Ridge models offer a more holistic view by incorporating Barrel% and K% alongside wOBA. The XGBoost model's poor performance (0.245 $R^2$) is explained by its "backwards" interpretation of HardHit% and K%. It appears to have overfitted to noise, whereas the other models maintained more logical relationships.
                """
            )
       
    
        
        
       

else:
    #what viewer sees before running predictions (landing page)
   
    st.info("Machine learning meets baseball. Predict 2025 player stats using 4 years of data, advanced metrics, and multiple ML models." \
    " Configure your prediction settings and click **Run Predictions** to start")
    
    # Show some quick stats about the dataset
    st.markdown("### Dataset Overview")
    
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    with quick_col1:
        st.metric("Total Player-Seasons", 1186) #hardcoded of num rows for prepared_data.csv
    with quick_col2:
        st.metric("Years Covered", "2020-2024")
    with quick_col3:
        st.metric("Models Available", "4")
    with quick_col4:
        st.metric("Stats Predicted", "4")