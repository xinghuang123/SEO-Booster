import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np  # For applying constraints
import openai
import base64


### 1. Declaring some useful functions.
### @st.cache_data is a decorator in Streamlit used to cache a function’s return value.
@st.cache_data
def process_rank_data(uploaded_file):
    """Process the uploaded keyword rank data file."""
    raw_df = pd.read_csv(uploaded_file)

    # Identify date columns
    date_columns = [col for col in raw_df.columns if '-' in col]

    # Standardize date abbreviations
    normalized_date_columns = [
        col.replace("Sept", "Sep") for col in date_columns
    ]

    # Rename columns in the DataFrame
    date_column_mapping = dict(zip(date_columns, normalized_date_columns))
    raw_df.rename(columns=date_column_mapping, inplace=True)

    # Parse dates
    parsed_dates = [datetime.strptime(col, '%b-%y').date() for col in normalized_date_columns]

    # Reshape the dataframe for easier analysis
    rank_df = raw_df.melt(
        ['Keyword'],
        normalized_date_columns,
        'Date',
        'Rank',
    )

    # Convert date column to datetime format
    rank_df['Date'] = rank_df['Date'].apply(lambda x: datetime.strptime(x, '%b-%y').date())

    return rank_df, parsed_dates

def predict_future_ranks(df, keywords):
    """Predict future ranks for the next 6 months using Linear Regression."""
    predictions = []
    for keyword in keywords:
        keyword_df = df[df['Keyword'] == keyword]
        keyword_df = keyword_df.sort_values('Date')

        # Prepare features and target
        keyword_df['Timestamp'] = keyword_df['Date'].map(datetime.toordinal)
        X = keyword_df[['Timestamp']]
        y = keyword_df['Rank']

        # Train-test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except ValueError:
            continue  # Skip keywords with insufficient data

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict future dates
        last_date = keyword_df['Date'].max()
        future_dates = [last_date + timedelta(days=30 * i) for i in range(1, 7)]
        future_timestamps = [[date.toordinal()] for date in future_dates]
        future_predictions = model.predict(future_timestamps)

        # Ensure predictions are positive
        future_predictions = np.maximum(future_predictions, 1)  # Cap predictions at 1

        # Append predictions
        for date, pred in zip(future_dates, future_predictions):
            predictions.append({'Keyword': keyword, 'Date': date, 'Rank': pred, 'Type': 'Prediction'})

    return pd.DataFrame(predictions)

### 2. Rendering the UI(User Interface)
### 2.1. Set the title and favicon that appear in the Browser`s tab bar.
st.set_page_config(
    page_title='SEO Booster',
    page_icon='rocket',
)

### 2.2 Set the logo, title and instruction that appears at the top of the page.
# Define a convert function
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Convert images to Base64
image1_base64 = get_image_base64("resources/NZSE-logo-blue.png")
image2_base64 = get_image_base64("resources/SEO-Booster-logo.png")

# Custom CSS
st.markdown(
    f"""
    <style>
    .logo-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .logo-container img {{
        max-width: 150px;
        height: auto;
    }}
    </style>
    <div class="logo-container">
        <img src="data:image/png;base64,{image2_base64}" alt="SEO Booster Logo">
        <img src="data:image/png;base64,{image1_base64}" alt="NZSE Logo">
    </div>
    """,
    unsafe_allow_html=True
)

### """...""" are multi-line string in Python
st.markdown(
    """ 
    <h1 style='text-align: center;'>SEO Booster 🚀</h1>

    **SEO** (**Search Engine Optimization**) is about helping search engines understand your content, and helping users find your site and make a decision about whether they should visit your site through a search engine.<br>

    **SEO Booster** empowers your SEO strategy with data-driven insights and AI-enhanced content creation! <br>

    All you need to do is just **four** steps:

    - **Upload Your Dataset**: Quickly integrate your data for analysis.
    - **Visualize SEO Trends**: Use interactive filters to explore keyword performance and ranking forecasts.
    - **Analyze Key Metrics**: Focus on search volume, TF-IDF scores, and more.
    - **Generate AI-Optimized Meta Description**: Receive keyword-based meta descriptions tailored to boost your SEO.
    """,
    unsafe_allow_html=True
)

### Add some spacing
''
''

### 2.3 Upload Your Dataset
st.header('• Upload Your Dataset', divider='gray')

# File uploader
uploaded_file = st.file_uploader("⚠️Please upload your dataset as a CSV file, ensuring it adheres to the data structure outlined in the template displayed below.", type=['csv'])

# Create a template DataFrame
def create_template_data():
    template_data = {
        'Keyword': ['keyword1', 'keyword2', 'keyword3'],
        'EndRank': [1, 2, 3],
        'EndClicks': [100, 200, 300],
        'SearchVolume': [1000, 1500, 2000],
        'description': ['example description', 'example description', 'example description'],
        'TFIDF_score': [0.8, 0.6, 0.7],
        'May-24':[1, 2, 3],
        'Jun-24':[2, 3, 1],
        'Jul-24':[3, 2, 1],
        'Aug-24':[1, 2, 3],
        'Sept-24':[2, 3, 1],
        'Oct-24':[3, 2, 1],
        'Nov-24':[1, 2, 3],
        'Dec-24':[2, 1, 5],
    }
    return pd.DataFrame(template_data)

# Generate template data
template_df = create_template_data()

# Display the template data
st.write("Template Dataset:")
st.dataframe(template_df)

# Generate a downloadable link for the template
csv_data = template_df.to_csv(index=False).encode('utf-8')  # Convert to CSV and encode as UTF-8

# Add a download link for the template
st.download_button(
    label="Download Template Dataset CSV file",
    data=csv_data,
    file_name="template_dataset.csv",
    mime="text/csv"
)

### Add some spacing
''
''

### 2.4 Visualize SEO Trends
st.header('• Visualize SEO Trends (Keyword Rankings over Time with Prediction)', divider='gray')

if uploaded_file is not None:
    rank_df, parsed_dates = process_rank_data(uploaded_file)

    min_date = min(parsed_dates)
    max_date = max(parsed_dates)

    # Date range slider
    st.markdown(
        """
        <h6 style='margin-bottom: 0;'>1. Select date range:</h6>
        """,
        unsafe_allow_html=True
    )

    from_date, to_date = st.slider(
        '',
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="MMM YYYY"
    )

    st.markdown(
        """
        <hr style='margin-top: 5px;' />
        """,
        unsafe_allow_html=True
    )

    # Keyword selector
    keywords = rank_df['Keyword'].unique()

    if not len(keywords):
        st.warning("Select at least one keyword")

    st.markdown(
        """
        <h6 style='margin-bottom: 0;'>2. Select keywords:</h6>
        """,
        unsafe_allow_html=True
    )

    selected_keywords = st.multiselect(
        '',
        keywords,
        keywords[:3]  # Select the first three keywords by default
    )

    st.markdown(
        """
        <hr style='margin-top: 5px;' />
        """,
        unsafe_allow_html=True
    )

    # Filter the data
    filtered_df = rank_df[
        (rank_df['Keyword'].isin(selected_keywords))
        & (rank_df['Date'] >= from_date)
        & (rank_df['Date'] <= to_date)
        ]
    filtered_df['Type'] = 'Historical'

    st.markdown(
        """
        <h6 style='margin-bottom: 0;'>3. Keyword Rankings over Time with Prediction:</h6>
        <br>
        """,
        unsafe_allow_html=True
    )

    # Predict future ranks
    future_df = predict_future_ranks(rank_df, selected_keywords)

    # Combine historical and future data
    combined_df = pd.concat([filtered_df, future_df], ignore_index=True)

    # Visualization
    import altair as alt

    highlight = alt.selection_single(
        fields=['Date'],
        nearest=True,
        on='mouseover',
        empty='none',
    )

    # Define a custom color palette for distinct line colors
    custom_colors = alt.Scale(
        domain=selected_keywords,
        range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    )

    base = alt.Chart(combined_df).encode(
        x=alt.X('Date:T', title='Date', axis=alt.Axis(format='%b-%y')),
        y=alt.Y('Rank:Q', title='Rank', scale=alt.Scale(reverse=True)),
        color=alt.Color('Keyword:N', scale=custom_colors, legend=alt.Legend(title="Keywords"))
    )

    points = base.mark_circle(size=65).encode(
        opacity=alt.condition(highlight, alt.value(1), alt.value(0.3)),
        strokeDash=alt.condition(alt.datum.Type == 'Prediction', alt.value([5, 5]), alt.value([0])),
        tooltip=['Keyword:N', 'Date:T', 'Rank:Q']
    ).transform_filter(
        alt.datum.Type == 'Prediction'
    ).add_selection(
        highlight
    )

    lines = base.mark_line().encode(
        size=alt.condition(highlight, alt.value(3), alt.value(1)),
        strokeDash=alt.condition(alt.datum.Type == 'Prediction', alt.value([5, 5]), alt.value([0]))
    )

    chart = lines + points

    st.altair_chart(chart.interactive(), use_container_width=True)

    # Add a legend for axes
    st.markdown(
        """
        Legend:  
        X-axis: Date  
        Y-axis: Rank (lower values indicate better rankings)
        """
    )

    st.markdown(
        """
        <hr style='margin-top: 5px;' />
        """,
        unsafe_allow_html=True
    )

else:
    st.warning("Please upload a CSV file to proceed.")

### Add some spacing
''
''

### 2.5 Analyze Key Metrics
# Set header
st.header('• Analyze Key Metrics', divider='gray')

# Function to add row-wise styling (alternating colors)
def highlight_rows(row):
    if row.name % 2 == 0:
        return ['background-color: #f7f7f7'] * len(row)
    else:
        return ['background-color: #ffffff'] * len(row)

if uploaded_file is not None:
    if selected_keywords:
        # Load and filter the metrics data
        metrics_df = pd.read_csv(uploaded_file)
        metrics_df = metrics_df[metrics_df['Keyword'].isin(selected_keywords)]

        # Apply styling using pandas Styler
        styled_df = (
            metrics_df[['Keyword', 'EndRank', 'EndClicks', 'SearchVolume', 'TFIDF_score', 'description']]
            .style.apply(highlight_rows, axis=1)  # Add alternating row colors
            .format(
                {
                    'EndRank': '{:.0f}',
                    'EndClicks': '{:.0f}',
                    'SearchVolume': '{:,}',
                    'TFIDF_score': '{:.2f}'
                }
            )
            .set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('font-size', '14px'), ('text-align', 'center')]},
                {'selector': 'td', 'props': [('font-size', '12px'), ('padding', '8px')]},
                {'selector': 'tr:hover', 'props': [('background-color', '#eaeaea')]}]
            )
        )

        # Display styled dataframe
        st.write(styled_df.to_html(), unsafe_allow_html=True)
    else:
        st.warning("No keywords selected. Please select keywords in the previous step.")
else:
    st.warning("Please upload a CSV file to proceed.")

### Add some spacing
''
''

### 2.6 OpenAI API Integration for Meta Description Generation
st.header('• Generate AI-Optimized Meta Description')

# Input fields for keywords and meta description
keywords = st.text_input("Enter three keywords separated by commas:")
current_meta = st.text_area("Enter the current meta description of your website:")

if st.button("Generate AI-Optimized Meta Description"):
    if keywords and current_meta:
        try:
            #client = OpenAI()

            # Prepare the prompt
            prompt = (
                f"Keywords: {keywords}\n"
                f"Current Meta Description: {current_meta}\n"
                f"Generate an AI-optimized meta description for better SEO performance."
            )

            # Call OpenAI API using ChatCompletion
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[
                    {"role": "system", "content": "You are an expert SEO assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )

            # Extract and display the generated meta description
            optimized_meta = response.choices[0].message.content.strip()
            st.subheader("AI-Optimized Meta Description")
            st.write(optimized_meta)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please provide both keywords and a current meta description.")

### Add some spacing
''
''

### 2.7 End of the UI(User Interface)
st.markdown(
    """ 
    <hr>
    <p style='text-align: center;'>😊 Thank you for using SEO Booster 🚀</p>
    <p style='text-align: center;'>By Xiufan Huang & Xing Huang</p>
    <p style='text-align: center;'>© 2025 All rights reserved</p>


    """,
    unsafe_allow_html=True
)

