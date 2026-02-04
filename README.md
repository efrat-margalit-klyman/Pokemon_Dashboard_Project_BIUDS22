# 🎮 Pokemon Data Dashboard

An interactive Streamlit dashboard for exploring and analyzing Pokemon statistics from the PokeAPI. This application provides comprehensive visualizations and correlation analysis between physical attributes and combat statistics for 1000 Pokemon.

## Project Description

This project fetches real-time Pokemon data from the [PokeAPI](https://pokeapi.co/) and presents it through an intuitive, interactive dashboard built with Streamlit. Users can explore Pokemon characteristics, filter by type, compare stats, and discover correlations between physical traits (height, weight) and combat abilities (attack, defense, speed, etc.).

### Key Features

- **Real-time Data Fetching**: Pulls data for up to 1000 Pokemon directly from PokeAPI
- **Interactive Data Explorer**: Filter Pokemon by primary and secondary types, view individual stats, and browse the complete dataset
- **Type Distribution Analysis**: Visualize how Pokemon are distributed across different types
- **Combat Stats Rankings**: Identify the strongest and weakest Pokemon based on total combat stats
- **Correlation Analysis**: Explore relationships between physical attributes and combat statistics with interactive heatmaps
- **Dual Visualization Modes**: Toggle between interactive Plotly charts and static Matplotlib/Seaborn visualizations

### Dashboard Pages

1. **Data Explorer**
   - Filter Pokemon by primary (type_1) and secondary (type_2) types
   - View type distribution charts
   - Compare stats across types with box plots
   - Inspect individual Pokemon statistics
   - See top 10 strongest and weakest Pokemon rankings
   - Browse and customize the data table display

2. **Correlation Analysis**
   - Full correlation heatmap of all numeric statistics
   - Physical traits vs. combat stats correlation matrix
   - Scatter plot grid showing relationships between height/weight and attack/defense
   - Statistical distributions for all key metrics
   - Summary of strongest positive and negative correlations

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Required Dependencies

Install all required packages using pip:

```bash
pip install requests pandas matplotlib seaborn plotly streamlit
```

Or create a `requirements.txt` file with:

```
requests
pandas
matplotlib
seaborn
plotly
streamlit
```

Then run:

```bash
pip install -r requirements.txt
```

## How to Use

### Running the Application

1. **Clone or download** the project files to your local machine

2. **Navigate** to the project directory in your terminal:
   ```bash
   cd path/to/project
   ```

3. **Execute the main file** using Streamlit:
   ```bash
   streamlit run main.py
   ```

4. **Wait for data loading**: On first launch, the application will fetch Pokemon data from the API. This may take a few minutes as it retrieves details for 1000 Pokemon. A progress bar will show the loading status.

5. **Explore the dashboard**: Once loaded, use the sidebar to navigate between pages and apply filters.

### File to Execute

```
main.py
```

This is the main entry point for the application. Run it with Streamlit as shown above.

### Navigation

- Use the **sidebar** on the left to switch between "Data Explorer" and "Correlation Analysis" pages
- In Data Explorer, use the **Type Filters** in the sidebar to filter Pokemon by their types
- Select visualization styles (Interactive/Static) using the radio buttons where available

## Data Processing

The application performs several data transformations:

- **Extracts** basic stats (name, height, weight, base experience)
- **Parses** Pokemon types (primary and secondary)
- **Computes** combat statistics (attack, defense, special-attack, special-defense, speed)
- **Derives** additional columns:
  - `total_combat_stats`: Sum of all combat statistics
  - `speed_category`: Categorizes Pokemon as Slow, Average, or Fast
  - `weight_class`: Categorizes as light, medium, or heavy
- **Filters outliers**: Removes extreme values (top 5% by weight and height) for cleaner visualizations

## Streamlit Deployment

Access the live deployed version of this dashboard:

**[Pokemon Data Dashboard on Streamlit Cloud](https://pokemondashboardclassprojectpython-biuds22.streamlit.app/)**


### Deploying Your Own Instance

1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Select your repository and `main.py` as the entry point
5. Click "Deploy"

## Project Structure

```
pokemon-dashboard/
│
├── main.py              # Main application file (execute this)
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies (optional)
```

## Usage Examples

### Filtering by Type
1. In the Data Explorer page, expand the sidebar
2. Select one or more primary types (e.g., "fire", "water")
3. Optionally select secondary types or "None" for single-type Pokemon
4. View the filtered results and statistics

### Analyzing Correlations
1. Navigate to the "Correlation Analysis" page
2. Examine the heatmaps to see which stats are most correlated
3. Use the scatter plots to visualize relationships between physical and combat stats
4. Review the correlation summary tables for quick insights

## License

This project is open source and available for educational and personal use.

## Acknowledgments

- Data provided by [PokeAPI](https://pokeapi.co/)
- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Plotly](https://plotly.com/), [Matplotlib](https://matplotlib.org/), and [Seaborn](https://seaborn.pydata.org/)

