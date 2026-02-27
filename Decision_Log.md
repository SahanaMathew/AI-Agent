# Decision Log - Monday.com Business Intelligence Agent

## 1. Architecture Decisions

### Tech Stack Selection
| Component | Choice | Rationale |
|-----------|--------|-----------|
| **LLM Provider** | Groq (Llama 3.3 70B) | Fast inference, free tier, excellent reasoning capabilities for business queries |
| **Frontend** | Streamlit | Rapid development, built-in chat UI, easy deployment to Streamlit Cloud |
| **Monday.com Integration** | Direct GraphQL API | Full control over queries, better error handling than MCP for production use |
| **Language** | Python 3.12 | Best ecosystem for LLM/data processing, extensive Monday.com community support |

### Why Not MCP?
While MCP (Model Context Protocol) was mentioned as a bonus, I chose direct API integration because:
1. **Reliability**: Direct API gives better error handling and retry logic
2. **Transparency**: Easier to show visible tool traces (assignment requirement)
3. **Flexibility**: Can customize data processing and filtering at query level
4. **Deployment**: Simpler deployment without MCP server dependencies

## 2. Data Handling Strategy

### Messy Data Challenges Identified
- **Deal Values**: 52% missing (181/346 records)
- **Closure Probability**: 75% missing (258/346 records)
- **Close Dates**: 92% missing (318/346 records)
- **Work Order Amounts**: Various currency formats and nulls

### Solutions Implemented
1. **Null Handling**: All aggregations use `pandas` with `skipna=True`
2. **Format Normalization**: Strip currency symbols, handle comma separators
3. **Data Quality Reporting**: Agent reports caveats when data gaps affect analysis
4. **Sector Standardization**: Map variations (e.g., "power line" → "Powerline")

## 3. Agent Design Decisions

### Tool Design
Three focused tools instead of generic "query board":
1. `get_deals_data` - Fetches all deals with processing
2. `get_work_orders_data` - Fetches all work orders with processing
3. `get_board_schema` - Returns column metadata for understanding structure

**Rationale**: Specific tools give the LLM clearer intent signals, improving accuracy.

### No Caching
Per requirements, every query triggers live API calls. No local caching or pre-loading.

### Conversation Context
Maintains last 10 messages for follow-up questions while preventing context overflow.

## 4. UI/UX Decisions

### Tool Trace Visibility
- Sidebar shows real-time API call traces
- Each trace shows: tool name, status, records fetched
- Color-coded: green (success), yellow (executing), red (error)

**Rationale**: Meets "visible action/tool-call trace" requirement while keeping main chat clean.

### Sample Questions
Pre-built buttons for common queries to help evaluators quickly test functionality.

## 5. Trade-offs Made

| Trade-off | Decision | Impact |
|-----------|----------|--------|
| Speed vs Accuracy | Fetch all records, filter in Python | Slower on large boards, but more accurate aggregations |
| Complexity vs Maintainability | Single-file agent vs microservices | Easier to understand and deploy |
| MCP vs Direct API | Direct API | Lost bonus points, gained reliability |

## 6. What I Would Improve With More Time

1. **Pagination**: Handle boards with >500 items
2. **Semantic Caching**: Cache schema/column info (not data) to reduce latency
3. **Charts**: Add Plotly visualizations for trends
4. **MCP Integration**: Add as optional alternative to direct API
5. **Testing**: Unit tests for data processor edge cases

## 7. Time Allocation

| Phase | Time Spent |
|-------|------------|
| Requirements Analysis | 30 min |
| Monday.com Setup | 30 min |
| API Client Development | 45 min |
| Data Processor | 45 min |
| LLM Agent | 1.5 hrs |
| Streamlit UI | 1 hr |
| Testing & Debugging | 1 hr |
| Documentation | 30 min |
| **Total** | ~6 hrs |
