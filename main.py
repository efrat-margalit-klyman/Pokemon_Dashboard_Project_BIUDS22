import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


API_BASE_URL = "https://pokeapi.co/api/v2"

PHYSICAL_STATS = ['height', 'weight']
COMBAT_STATS = ['attack', 'defense', 'special-attack', 'special-defense', 'speed']
ALL_NUMERIC = ['height', 'weight', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']


def fetch_pokemon_list(limit=1000):
    url = f"{API_BASE_URL}/pokemon?limit={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['results']
    else:
        raise Exception(f"Failed to fetch Pokemon list: {response.status_code}")


def fetch_pokemon_details(pokemon_url):
    response = requests.get(pokemon_url)
    if response.status_code == 200:
        return response.json()
    return None


def fetch_all_pokemon_data(limit=1000, progress_callback=None):
    pokemon_list = fetch_pokemon_list(limit)
    full_pokemon_data = []

    for i in range(len(pokemon_list)):
        current_pokemon = pokemon_list[i]
        url_to_fetch = current_pokemon.get('url')
        data = fetch_pokemon_details(url_to_fetch)

        if data:
            full_pokemon_data.append(data)

        if progress_callback:
            progress_callback(i + 1, len(pokemon_list))

    pokemon_df = pd.DataFrame(full_pokemon_data)
    return pokemon_df


def extract_basic_stats(df):
    return df[['name', 'height', 'weight', 'base_experience']].copy()


def extract_types(df):
    type1_values = []
    type2_values = []

    for types_list in df['types']:
        t1 = types_list[0]['type']['name']
        type1_values.append(t1)

        if len(types_list) > 1:
            t2 = types_list[1]['type']['name']
        else:
            t2 = None

        type2_values.append(t2)

    return type1_values, type2_values


def extract_combat_stats(df):
    attack_list = []
    defense_list = []
    sp_attack_list = []
    sp_defense_list = []
    speed_list = []

    for stats_list in df['stats']:
        temp_stats = {}
        for stat_info in stats_list:
            stat_name = stat_info['stat']['name']
            stat_value = stat_info['base_stat']
            temp_stats[stat_name] = stat_value

        attack_list.append(temp_stats.get('attack'))
        defense_list.append(temp_stats.get('defense'))
        sp_attack_list.append(temp_stats.get('special-attack'))
        sp_defense_list.append(temp_stats.get('special-defense'))
        speed_list.append(temp_stats.get('speed'))

    return {
        'attack': attack_list,
        'defense': defense_list,
        'special-attack': sp_attack_list,
        'special-defense': sp_defense_list,
        'speed': speed_list
    }


def create_clean_dataframe(df):
    clean_df = extract_basic_stats(df)

    type1_values, type2_values = extract_types(df)
    clean_df['type_1'] = type1_values
    clean_df['type_2'] = type2_values

    combat_stats = extract_combat_stats(df)
    for stat_name, values in combat_stats.items():
        clean_df[stat_name] = values

    clean_df['speed_category'] = pd.cut(
        clean_df['speed'],
        bins=[0, 60, 90, 200],
        labels=['Slow', 'Average', 'Fast']
    )

    return clean_df


def add_derived_columns(df):
    result_df = df.copy()

    combat_cols = ['attack', 'defense', 'special-attack', 'special-defense', 'speed']
    result_df['total_combat_stats'] = result_df[combat_cols].sum(axis=1)

    combat_cols_no_speed = ['attack', 'defense', 'special-attack', 'special-defense']
    result_df['total_combat_stats_no_speed'] = result_df[combat_cols_no_speed].sum(axis=1)

    weight_bins = [0, 100, 500, float('inf')]
    weight_labels = ['light', 'medium', 'heavy']
    result_df['weight_class'] = pd.cut(
        result_df['weight'],
        bins=weight_bins,
        labels=weight_labels
    )

    return result_df


def filter_outliers(df, weight_percentile=0.95, height_percentile=0.95):
    weight_threshold = df['weight'].quantile(weight_percentile)
    height_threshold = df['height'].quantile(height_percentile)

    filtered_df = df[
        (df['weight'] <= weight_threshold) &
        (df['height'] <= height_threshold)
    ].copy()

    return filtered_df


def get_unique_types(df, type_column):
    types = df[type_column].dropna().unique().tolist()
    return sorted(types)


def filter_by_type(df, type_1=None, type_2=None):
    filtered_df = df.copy()

    if type_1 and len(type_1) > 0:
        filtered_df = filtered_df[filtered_df['type_1'].isin(type_1)]

    if type_2 and len(type_2) > 0:
        if 'None' in type_2:
            type_2_filtered = [t for t in type_2 if t != 'None']
            filtered_df = filtered_df[
                (filtered_df['type_2'].isin(type_2_filtered)) |
                (filtered_df['type_2'].isna())
            ]
        else:
            filtered_df = filtered_df[filtered_df['type_2'].isin(type_2)]

    return filtered_df


def create_type_distribution_chart(df, type_column='type_1'):
    type_counts = df[type_column].value_counts().reset_index()
    type_counts.columns = [type_column, 'count']

    fig = px.bar(
        type_counts,
        x=type_column,
        y='count',
        title=f'Pokemon Distribution by {type_column.replace("_", " ").title()}',
        color=type_column,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    return fig


def create_box_plot_by_type(df, stat_column, type_column='type_1'):
    fig = px.box(
        df,
        x=type_column,
        y=stat_column,
        color=type_column,
        title=f'{stat_column.replace("_", " ").title()} Distribution by Type',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    return fig


def plot_stats_histograms(df, stats=None):
    if stats is None:
        stats = ['weight', 'height', 'speed', 'attack', 'defense']

    n_stats = len(stats)
    n_cols = 3
    n_rows = (n_stats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))
    axes = axes.flatten() if n_stats > 1 else [axes]

    colors = sns.color_palette('Set2', n_stats)

    for i, stat in enumerate(stats):
        if stat in df.columns:
            sns.histplot(data=df, x=stat, kde=True, color=colors[i], ax=axes[i])
            axes[i].set_title(f'{stat.title()} Distribution')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig


def plot_total_combat_stats_histogram(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['total_combat_stats'], bins=20, color='turquoise', edgecolor='black')
    ax.set_title('Distribution of Total Combat Stats', fontsize=14, fontweight='bold')
    ax.set_xlabel('Total Combat Power', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_top_n_pokemon(df, n=10, strongest=True):
    if strongest:
        top_n = df.nlargest(n, 'total_combat_stats').sort_values('total_combat_stats', ascending=True)
        color = 'purple'
        title = f'Top {n} Strongest Pokemon by Total Combat Stats'
    else:
        top_n = df.nsmallest(n, 'total_combat_stats').sort_values('total_combat_stats', ascending=False)
        color = 'pink'
        title = f'Top {n} Weakest Pokemon by Total Combat Stats'

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.barh(range(len(top_n)), top_n['total_combat_stats'], color=color, edgecolor='black', linewidth=1.2)

    ax.set_yticks(range(len(top_n)))
    ax.set_yticklabels(top_n['name'].str.title(), fontsize=11)

    for i, val in enumerate(top_n['total_combat_stats']):
        ax.text(val + 5, i, f'{int(val)} pts', va='center')

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Total Combat Power', fontsize=13, fontweight='bold')
    ax.set_ylabel('Pokemon Name', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def calculate_correlation_matrix(df, columns=None):
    if columns is None:
        columns = ALL_NUMERIC
    available_cols = [col for col in columns if col in df.columns]
    return df[available_cols].corr()


def create_full_correlation_heatmap(df):
    corr_matrix = calculate_correlation_matrix(df)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
                vmin=-1, vmax=1, cmap='RdYlGn_r', ax=ax)

    ax.set_title('Pokemon Statistics Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def create_correlation_heatmap_plotly(df):
    corr_matrix = calculate_correlation_matrix(df)

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdYlGn_r',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(title='Correlation')
    ))

    fig.update_layout(
        title='Pokemon Statistics Correlation Matrix',
        xaxis_tickangle=-45
    )
    return fig


def create_physical_combat_correlation_heatmap(df):
    all_vars = PHYSICAL_STATS + COMBAT_STATS
    corr_matrix = df[all_vars].corr()

    physical_combat_corr = corr_matrix.loc[COMBAT_STATS, PHYSICAL_STATS]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(physical_combat_corr,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                center=0,
                linewidths=2,
                cbar_kws={'label': 'Correlation Coefficient'},
                vmin=-0.5, vmax=0.5,
                ax=ax)

    ax.set_title('Physical Traits → Combat Stats Correlation Matrix\n'
                 'Does Size Affect Fighting Ability?',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Physical Traits', fontsize=13, fontweight='bold')
    ax.set_ylabel('Combat Stats', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def create_physical_combat_correlation_plotly(df):
    all_vars = PHYSICAL_STATS + COMBAT_STATS
    corr_matrix = df[all_vars].corr()

    physical_combat_corr = corr_matrix.loc[COMBAT_STATS, PHYSICAL_STATS]

    fig = go.Figure(data=go.Heatmap(
        z=physical_combat_corr.values,
        x=physical_combat_corr.columns,
        y=physical_combat_corr.index,
        colorscale='RdYlGn',
        zmid=0,
        zmin=-0.5,
        zmax=0.5,
        text=physical_combat_corr.round(3).values,
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverongaps=False,
        colorbar=dict(title='Correlation')
    ))

    fig.update_layout(
        title='Physical Traits → Combat Stats Correlation<br>'
              '<sub>Does Size Affect Fighting Ability?</sub>',
        xaxis_title='Physical Traits',
        yaxis_title='Combat Stats',
        xaxis_tickangle=0
    )
    return fig


def create_physical_combat_scatter_grid(df):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Weight vs Attack', 'Weight vs Defense',
            'Height vs Attack', 'Height vs Defense'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    color_map = {'Slow': '#ff7f0e', 'Average': '#2ca02c', 'Fast': '#1f77b4'}

    for category in ['Slow', 'Average', 'Fast']:
        subset = df[df['speed_category'] == category]
        color = color_map.get(category, '#1f77b4')

        fig.add_trace(
            go.Scatter(x=subset['weight'], y=subset['attack'],
                       mode='markers', name=category,
                       marker=dict(color=color, size=8, opacity=0.7),
                       legendgroup=category, showlegend=True),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=subset['weight'], y=subset['defense'],
                       mode='markers', name=category,
                       marker=dict(color=color, size=8, opacity=0.7),
                       legendgroup=category, showlegend=False),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=subset['height'], y=subset['attack'],
                       mode='markers', name=category,
                       marker=dict(color=color, size=8, opacity=0.7),
                       legendgroup=category, showlegend=False),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=subset['height'], y=subset['defense'],
                       mode='markers', name=category,
                       marker=dict(color=color, size=8, opacity=0.7),
                       legendgroup=category, showlegend=False),
            row=2, col=2
        )

    fig.update_xaxes(title_text='Weight', row=1, col=1)
    fig.update_xaxes(title_text='Weight', row=1, col=2)
    fig.update_xaxes(title_text='Height', row=2, col=1)
    fig.update_xaxes(title_text='Height', row=2, col=2)
    fig.update_yaxes(title_text='Attack', row=1, col=1)
    fig.update_yaxes(title_text='Defense', row=1, col=2)
    fig.update_yaxes(title_text='Attack', row=2, col=1)
    fig.update_yaxes(title_text='Defense', row=2, col=2)

    fig.update_layout(
        title_text='Physical Stats vs Combat Stats (by Speed Category)',
        height=700,
        legend_title_text='Speed Category'
    )
    return fig


def get_strongest_correlations(df, n=5):
    corr_matrix = calculate_correlation_matrix(df)

    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Variable 1': corr_matrix.columns[i],
                'Variable 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

    corr_df = pd.DataFrame(corr_pairs)
    corr_df = corr_df.sort_values('Correlation', ascending=False)

    top_positive = corr_df.head(n)
    top_negative = corr_df.tail(n).sort_values('Correlation')

    return top_positive, top_negative


def fetch_and_process_pokemon_data(limit=1000, progress_callback=None):
    raw_df = fetch_all_pokemon_data(limit=limit, progress_callback=progress_callback)
    clean_df = create_clean_dataframe(raw_df)
    clean_df = add_derived_columns(clean_df)
    filtered_df = filter_outliers(clean_df)
    return filtered_df


def init_session_state():
    if 'pokemon_df' not in st.session_state:
        st.session_state.pokemon_df = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False


def create_sidebar_filters(df):
    st.sidebar.header("Type Filters")

    type_1_options = get_unique_types(df, 'type_1')
    type_2_list = df['type_2'].dropna().unique().tolist()
    type_2_options = ['None'] + sorted(type_2_list)

    st.sidebar.subheader("Primary Type (type_1)")
    selected_type_1 = st.sidebar.multiselect(
        "Select Primary Type(s)",
        options=type_1_options,
        default=[]
    )

    st.sidebar.subheader("Secondary Type (type_2)")
    selected_type_2 = st.sidebar.multiselect(
        "Select Secondary Type(s)",
        options=type_2_options,
        default=[]
    )

    return selected_type_1, selected_type_2


def create_data_explorer_page(df):
    st.title("🔍 Pokemon Data Explorer")
    st.markdown("Explore Pokemon data with interactive filters")

    selected_type_1, selected_type_2 = create_sidebar_filters(df)
    filtered_df = filter_by_type(df, selected_type_1, selected_type_2)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Pokemon", len(df))
    with col2:
        st.metric("Filtered Pokemon", len(filtered_df))

    st.markdown("---")
    st.subheader("Distribution by Primary Type")

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        if len(filtered_df) > 0:
            fig_type_dist = create_type_distribution_chart(filtered_df, 'type_1')
            st.plotly_chart(fig_type_dist, use_container_width=True)

    with viz_col2:
        if len(filtered_df) > 0:
            stat_to_compare = st.selectbox(
                "Select stat to compare by type",
                options=['attack', 'defense', 'speed', 'total_combat_stats'],
                index=0
            )
            fig_box = create_box_plot_by_type(filtered_df, stat_to_compare)
            st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    st.subheader("Individual Pokemon Stats")

    if len(filtered_df) > 0:
        selected_pokemon = st.selectbox(
            "Select a Pokemon to view detailed stats",
            options=filtered_df['name'].tolist()
        )

        if selected_pokemon:
            pokemon_stats = filtered_df[filtered_df['name'] == selected_pokemon].iloc[0]
            stat_cols = st.columns(5)

            with stat_cols[0]:
                st.metric("Height", int(pokemon_stats['height']))
            with stat_cols[1]:
                st.metric("Weight", int(pokemon_stats['weight']))
            with stat_cols[2]:
                st.metric("Speed", int(pokemon_stats['speed']))
            with stat_cols[3]:
                st.metric("Attack", int(pokemon_stats['attack']))
            with stat_cols[4]:
                st.metric("Defense", int(pokemon_stats['defense']))

    st.markdown("---")
    st.subheader("Top Pokemon Rankings")

    rank_col1, rank_col2 = st.columns(2)

    with rank_col1:
        st.markdown("**Top 10 Strongest**")
        fig_top = plot_top_n_pokemon(filtered_df, n=10, strongest=True)
        st.pyplot(fig_top)

    with rank_col2:
        st.markdown("**Top 10 Weakest**")
        fig_weak = plot_top_n_pokemon(filtered_df, n=10, strongest=False)
        st.pyplot(fig_weak)

    st.markdown("---")
    st.subheader("Filtered Pokemon Data")

    all_columns = filtered_df.columns.tolist()
    default_columns = ['name', 'type_1', 'type_2', 'attack', 'defense',
                       'special-attack', 'special-defense', 'speed', 'total_combat_stats']
    available_defaults = [col for col in default_columns if col in all_columns]

    selected_columns = st.multiselect(
        "Select columns to display",
        options=all_columns,
        default=available_defaults
    )

    if selected_columns:
        st.dataframe(
            filtered_df[selected_columns].reset_index(drop=True),
            use_container_width=True,
            height=400
        )
    else:
        st.warning("Please select at least one column to display.")


def create_correlation_page(df):
    st.title("Physical vs Combat Stats Correlation Analysis")
    st.markdown("Explore correlations between physical attributes and combat statistics")

    st.markdown("---")
    st.subheader("Full Correlation Heatmap")

    heatmap_type = st.radio(
        "Select heatmap style",
        options=['Interactive (Plotly)', 'Static (Matplotlib)'],
        horizontal=True
    )

    if heatmap_type == 'Interactive (Plotly)':
        fig_heatmap = create_correlation_heatmap_plotly(df)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        fig_heatmap = create_full_correlation_heatmap(df)
        st.pyplot(fig_heatmap)

    st.markdown("---")
    st.subheader("Physical Traits → Combat Stats Correlation")
    st.markdown("*Does size affect fighting ability?*")

    phys_combat_type = st.radio(
        "Select visualization style",
        options=['Interactive (Plotly)', 'Static (Matplotlib)'],
        horizontal=True,
        key='phys_combat_radio'
    )

    if phys_combat_type == 'Interactive (Plotly)':
        fig_phys_combat = create_physical_combat_correlation_plotly(df)
        st.plotly_chart(fig_phys_combat, use_container_width=True)
    else:
        fig_phys_combat = create_physical_combat_correlation_heatmap(df)
        st.pyplot(fig_phys_combat)

    st.markdown("---")
    st.subheader("Physical vs Combat Scatter Plots")

    fig_grid = create_physical_combat_scatter_grid(df)
    st.plotly_chart(fig_grid, use_container_width=True)

    st.markdown("---")
    st.subheader("Stats Distributions")

    fig_histograms = plot_stats_histograms(df)
    st.pyplot(fig_histograms)

    st.markdown("---")
    st.subheader("Total Combat Stats Distribution")

    fig_total_combat = plot_total_combat_stats_histogram(df)
    st.pyplot(fig_total_combat)

    st.markdown("---")
    st.subheader("Correlation Summary Statistics")

    top_positive, top_negative = get_strongest_correlations(df, n=5)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 5 Positive Correlations:**")
        st.dataframe(top_positive, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Top 5 Negative Correlations:**")
        st.dataframe(top_negative, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="Pokemon Data Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()

    st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("🎮 Pokemon Dashboard")
    st.sidebar.markdown("---")

    if not st.session_state.data_loaded:
        with st.spinner("Fetching Pokemon data from API... This may take a few minutes..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"Fetching Pokemon {current}/{total}...")

            df = fetch_and_process_pokemon_data(limit=1000, progress_callback=update_progress)
            st.session_state.pokemon_df = df
            st.session_state.data_loaded = True

            progress_bar.empty()
            status_text.empty()
            st.rerun()

    df = st.session_state.pokemon_df

    page = st.sidebar.radio(
        "Navigation",
        options=["Data Explorer", "Correlation Analysis"],
        index=0
    )

    if page == "Data Explorer":
        create_data_explorer_page(df)
    elif page == "Correlation Analysis":
        create_correlation_page(df)


if __name__ == "__main__":
    main()
