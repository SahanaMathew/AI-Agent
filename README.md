# Monday.com Business Intelligence Agent

A conversational AI agent that answers founder-level business intelligence queries by fetching live data from Monday.com boards.

## Features

- **Live Monday.com Integration**: Queries data in real-time via GraphQL API (no caching)
- **Natural Language Understanding**: Interprets business questions and translates to API calls
- **Data Resilience**: Handles missing values, normalizes inconsistent formats
- **Visible Tool Traces**: Shows API calls being made in real-time
- **Cross-board Analysis**: Queries both Deals and Work Orders boards
- **Conversational Context**: Supports follow-up questions

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (Llama 3.3 70B)
- **Data Source**: Monday.com GraphQL API
- **Language**: Python 3.12

## Project Structure

```
monday-bi-agent/
├── app.py                 # Streamlit main application
├── agent.py               # LLM agent with tool calling
├── monday_client.py       # Monday.com GraphQL API client
├── data_processor.py      # Data cleaning and normalization
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in repo)
├── .gitignore
├── .streamlit/
│   └── config.toml        # Streamlit configuration
└── README.md
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd monday-bi-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file with:

```
MONDAY_API_TOKEN=your_monday_api_token
GROQ_API_KEY=your_groq_api_key
DEALS_BOARD_ID=your_deals_board_id
WORK_ORDERS_BOARD_ID=your_work_orders_board_id
```

### 4. Run the application

```bash
streamlit run app.py
```

## Monday.com Board Setup

Import the provided Excel files into Monday.com:
1. `Deal funnel Data.xlsx` → Create "Deals" board
2. `Work_Order_Tracker Data.xlsx` → Create "Work Orders" board

## Sample Queries

- "How's our pipeline looking?"
- "What's the total deal value by sector?"
- "Show me deals with high closure probability"
- "What's our billing status on work orders?"
- "Which sectors have the most open deals?"
- "What's the collection status for work orders?"

## Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add secrets in Streamlit Cloud dashboard:
   - `MONDAY_API_TOKEN`
   - `GROQ_API_KEY`
   - `DEALS_BOARD_ID`
   - `WORK_ORDERS_BOARD_ID`

## Data Quality Handling

The agent handles:
- Missing/null values in deal amounts, dates, probabilities
- Inconsistent sector naming conventions
- Various date formats
- Partial financial data in work orders

Quality warnings are displayed when significant data gaps are detected.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  STREAMLIT FRONTEND                      │
│  • Chat Interface                                        │
│  • Tool Call Trace Panel                                │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              GROQ LLM (Llama 3.3 70B)                    │
│  • Query Understanding                                   │
│  • Tool Selection                                        │
│  • Response Generation                                   │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   TOOL LAYER                             │
│  • get_deals_data()                                      │
│  • get_work_orders_data()                               │
│  • get_board_schema()                                   │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              DATA PROCESSOR                              │
│  • Null handling                                        │
│  • Format normalization                                 │
│  • Quality assessment                                   │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│          MONDAY.COM GraphQL API                          │
│  • Deals Board                                          │
│  • Work Orders Board                                    │
└─────────────────────────────────────────────────────────┘
```

## License

MIT
