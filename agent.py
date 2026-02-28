"""
AI Agent Module
Handles query understanding, tool calling, and response generation using Groq LLM
"""

import os
import json
from typing import List, Dict, Any, Optional, Generator
from groq import Groq
from dotenv import load_dotenv
import pandas as pd

from monday_client import MondayClient
from data_processor import DataProcessor, get_data_quality_summary

load_dotenv()


def format_indian_currency(value: float) -> str:
    """
    Format a number in Indian currency style with complete breakdown.
    1 Crore = 10,000,000 (1 followed by 7 zeros)
    1 Lakh = 100,000 (1 followed by 5 zeros)
    1 Thousand = 1,000
    Shows exact value: X Cr Y L Z K (e.g., ₹10,73,89,777)
    """
    if value is None or pd.isna(value):
        return "N/A"

    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    # Round to nearest rupee for display
    abs_value = round(abs_value)

    # Format in Indian numbering system (XX,XX,XX,XXX)
    if abs_value >= 10000000:  # >= 1 Crore
        crores = int(abs_value // 10000000)
        remaining = int(abs_value % 10000000)
        lakhs = remaining // 100000
        thousands = (remaining % 100000) // 1000
        hundreds = remaining % 1000

        # Build Indian format: Cr,LL,TT,HHH
        formatted = f"{sign}₹{crores},{lakhs:02d},{thousands:02d},{hundreds:03d}"
    elif abs_value >= 100000:  # >= 1 Lakh
        lakhs = int(abs_value // 100000)
        remaining = int(abs_value % 100000)
        thousands = remaining // 1000
        hundreds = remaining % 1000

        formatted = f"{sign}₹{lakhs},{thousands:02d},{hundreds:03d}"
    elif abs_value >= 1000:
        thousands = int(abs_value // 1000)
        hundreds = int(abs_value % 1000)
        formatted = f"{sign}₹{thousands},{hundreds:03d}"
    else:
        formatted = f"{sign}₹{int(abs_value)}"

    return formatted


def format_value_dict(value_dict: dict) -> dict:
    """Format all values in a dictionary to Indian currency style."""
    return {k: format_indian_currency(v) for k, v in value_dict.items()}


# Define tools for the agent
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_deals_data",
            "description": "Fetch all deals from the Monday.com Deals board. Use this for questions about deals, pipeline, revenue potential, deal stages, closure probability, sectors, and sales performance.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_work_orders_data",
            "description": "Fetch all work orders from the Monday.com Work Orders board. Use this for questions about work orders, projects, billing, invoices, collections, execution status, and operational performance.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_board_schema",
            "description": "Get the column schema for a specific board. Use this to understand what data fields are available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "board": {
                        "type": "string",
                        "enum": ["deals", "work_orders"],
                        "description": "Which board to get schema for"
                    }
                },
                "required": ["board"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are a Business Intelligence assistant for a company that uses Monday.com to track Deals and Work Orders. Your role is to answer founder-level business questions by querying live data from Monday.com.

## Your Capabilities:
1. **Deals Board**: Contains sales pipeline data including deal names, values, stages, closure probability, sectors (Mining, Powerline, Renewables, etc.), and dates.
2. **Work Orders Board**: Contains project execution data including work order status, billing amounts, collection status, and operational metrics.

## Guidelines:
- Always fetch LIVE data using the provided tools - never make assumptions about the data
- ALWAYS use the pre-computed summaries in the tool response (like "open_deals_by_sector", "delayed_work_orders", "value_by_sector", etc.) - these are the ACCURATE aggregations calculated from ALL records
- Provide specific numbers and percentages when possible
- For ambiguous questions, ask clarifying questions
- Support follow-up questions by remembering context

## Response Format:
- Start with a DIRECT, CONCISE answer to the question
- Include relevant metrics and numbers in a clear format (use tables or bullet points)
- Keep responses focused and executive-friendly - founders want quick insights, not data dumps

## CRITICAL - Currency Formatting (MUST FOLLOW):
- All values are in Indian Rupees (₹)
- **ALWAYS use the "_formatted" fields** from the summary data - these contain EXACT pre-calculated values
- NEVER convert or calculate currency values yourself - you WILL make errors
- Copy the formatted values EXACTLY as they appear (e.g., "₹11,74,464" not "₹0.01 Cr")
- The formatted values use Indian numbering: ₹Cr,LL,TT,HHH (Crores, Lakhs, Thousands, Hundreds)
- Examples of CORRECT usage:
  * If "value_by_sector_formatted": {"Mining": "₹45,48,38,417"} → use "₹45,48,38,417"
  * If "value_by_sector_formatted": {"Construction": "₹11,74,464"} → use "₹11,74,464"
  * If "total_pipeline_value_formatted": "₹230,55,18,041" → use "₹230,55,18,041"
- WRONG: Converting ₹1174464 to "₹0.01 Cr" - NEVER DO THIS

## Data Quality Reporting (IMPORTANT - Keep it brief):
- Only mention data quality issues that are RELEVANT to the specific question asked
- For value calculations, briefly note: "Based on X records with data"
- Do NOT list every missing column - only mention if it directly affects the answer
- Example good caveat: "Note: 19 work orders missing end dates were excluded from delay calculation"
- Example bad caveat: Listing 20+ columns with missing values
"""


class BIAgent:
    """Business Intelligence Agent powered by Groq LLM"""

    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.monday_client = MondayClient()
        self.data_processor = DataProcessor()
        self.conversation_history: List[Dict] = []
        self.tool_traces: List[Dict] = []

    def _execute_tool(self, tool_name: str, tool_args: Dict) -> Dict[str, Any]:
        """Execute a tool and return results"""
        trace = {
            "tool": tool_name,
            "args": tool_args,
            "status": "executing"
        }
        self.tool_traces.append(trace)

        try:
            if tool_name == "get_deals_data":
                items = self.monday_client.get_deals_data()
                df, quality_report = self.data_processor.process_deals_data(items)
                trace["status"] = "success"
                trace["records_fetched"] = len(df)
                return {
                    "data": df.to_dict(orient='records') if not df.empty else [],
                    "quality_report": quality_report,
                    "summary": self._generate_deals_summary(df)
                }

            elif tool_name == "get_work_orders_data":
                items = self.monday_client.get_work_orders_data()
                df, quality_report = self.data_processor.process_work_orders_data(items)
                trace["status"] = "success"
                trace["records_fetched"] = len(df)
                return {
                    "data": df.to_dict(orient='records') if not df.empty else [],
                    "quality_report": quality_report,
                    "summary": self._generate_work_orders_summary(df)
                }

            elif tool_name == "get_board_schema":
                board = tool_args.get("board", "deals")
                if board == "deals":
                    columns = self.monday_client.get_deals_columns()
                else:
                    columns = self.monday_client.get_work_orders_columns()
                trace["status"] = "success"
                return {"columns": columns}

            else:
                trace["status"] = "error"
                trace["error"] = f"Unknown tool: {tool_name}"
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            trace["status"] = "error"
            trace["error"] = str(e)
            return {"error": str(e)}

    def _generate_deals_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive summary statistics for deals data"""
        if df.empty:
            return {"message": "No deals data available"}

        summary = {
            "total_deals": len(df),
        }

        # Find key columns
        value_col = next((col for col in df.columns if 'value' in col.lower()), None)
        status_col = next((col for col in df.columns if col.lower() == 'deal status'), None)
        sector_col = next((col for col in df.columns if 'sector' in col.lower()), None)
        stage_col = next((col for col in df.columns if 'stage' in col.lower()), None)
        prob_col = next((col for col in df.columns if 'probability' in col.lower()), None)
        owner_col = next((col for col in df.columns if 'owner' in col.lower()), None)

        # Value metrics
        if value_col:
            numeric_values = pd.to_numeric(df[value_col], errors='coerce')
            total_pipeline = float(numeric_values.sum())
            avg_deal = float(numeric_values.mean()) if numeric_values.notna().any() else 0
            summary["total_pipeline_value"] = total_pipeline
            summary["total_pipeline_value_formatted"] = format_indian_currency(total_pipeline)
            summary["avg_deal_value"] = avg_deal
            summary["avg_deal_value_formatted"] = format_indian_currency(avg_deal)
            summary["deals_with_value"] = int(numeric_values.notna().sum())

        # Status breakdown
        if status_col:
            summary["status_breakdown"] = df[status_col].value_counts().to_dict()

        # Sector breakdown
        if sector_col:
            summary["sector_breakdown"] = df[sector_col].value_counts().to_dict()

        # Stage breakdown
        if stage_col:
            summary["stage_breakdown"] = df[stage_col].value_counts().to_dict()

        # Probability breakdown
        if prob_col:
            summary["probability_breakdown"] = df[prob_col].value_counts().to_dict()

        # Owner breakdown
        if owner_col:
            summary["owner_breakdown"] = df[owner_col].value_counts().to_dict()

        # CROSS-TABULATED DATA - Critical for accurate answers
        # Deals by status and sector
        if status_col and sector_col:
            cross_tab = df.groupby([status_col, sector_col]).size().unstack(fill_value=0)
            summary["deals_by_status_and_sector"] = cross_tab.to_dict()

        # Deals by status and stage
        if status_col and stage_col:
            cross_tab = df.groupby([status_col, stage_col]).size().unstack(fill_value=0)
            summary["deals_by_status_and_stage"] = cross_tab.to_dict()

        # Value by sector
        if value_col and sector_col:
            value_by_sector = df.groupby(sector_col)[value_col].apply(
                lambda x: pd.to_numeric(x, errors='coerce').sum()
            ).to_dict()
            summary["value_by_sector"] = value_by_sector
            summary["value_by_sector_formatted"] = format_value_dict(value_by_sector)

        # Value by status
        if value_col and status_col:
            value_by_status = df.groupby(status_col)[value_col].apply(
                lambda x: pd.to_numeric(x, errors='coerce').sum()
            ).to_dict()
            summary["value_by_status"] = value_by_status
            summary["value_by_status_formatted"] = format_value_dict(value_by_status)

        # Deals by probability and sector
        if prob_col and sector_col:
            cross_tab = df.groupby([prob_col, sector_col]).size().unstack(fill_value=0)
            summary["deals_by_probability_and_sector"] = cross_tab.to_dict()

        # Open deals specifically (commonly asked)
        if status_col:
            open_deals = df[df[status_col] == 'Open']
            summary["open_deals_count"] = len(open_deals)
            if sector_col:
                summary["open_deals_by_sector"] = open_deals[sector_col].value_counts().to_dict()
            if stage_col:
                summary["open_deals_by_stage"] = open_deals[stage_col].value_counts().to_dict()
            if prob_col:
                summary["open_deals_by_probability"] = open_deals[prob_col].value_counts().to_dict()
            if value_col:
                open_values = pd.to_numeric(open_deals[value_col], errors='coerce')
                open_total = float(open_values.sum())
                summary["open_deals_total_value"] = open_total
                summary["open_deals_total_value_formatted"] = format_indian_currency(open_total)

        # Won deals
        if status_col:
            won_deals = df[df[status_col] == 'Won']
            summary["won_deals_count"] = len(won_deals)
            if sector_col:
                summary["won_deals_by_sector"] = won_deals[sector_col].value_counts().to_dict()
            if value_col:
                won_values = pd.to_numeric(won_deals[value_col], errors='coerce')
                won_total = float(won_values.sum())
                summary["won_deals_total_value"] = won_total
                summary["won_deals_total_value_formatted"] = format_indian_currency(won_total)

        # DATA QUALITY INFO - Critical for transparent reporting
        summary["data_quality"] = {
            "total_records": len(df),
            "missing_values": {}
        }
        for col in df.columns:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                null_pct = round((null_count / len(df)) * 100, 1)
                summary["data_quality"]["missing_values"][col] = {
                    "count": null_count,
                    "percentage": f"{null_pct}%"
                }

        # Key caveats for the LLM to mention
        caveats = []
        if value_col:
            value_nulls = int(pd.to_numeric(df[value_col], errors='coerce').isna().sum())
            if value_nulls > 0:
                caveats.append(f"Deal values: {value_nulls}/{len(df)} records missing ({round(value_nulls/len(df)*100,1)}%)")
        if prob_col:
            prob_nulls = int(df[prob_col].isna().sum())
            if prob_nulls > 0:
                caveats.append(f"Closure probability: {prob_nulls}/{len(df)} records missing ({round(prob_nulls/len(df)*100,1)}%)")
        summary["data_quality"]["key_caveats"] = caveats

        return summary

    def _generate_work_orders_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive summary statistics for work orders data"""
        if df.empty:
            return {"message": "No work orders data available"}

        summary = {
            "total_work_orders": len(df),
            "columns_available": list(df.columns)
        }

        # Find key columns
        exec_status_col = next((col for col in df.columns if 'execution' in col.lower() and 'status' in col.lower()), None)
        sector_col = next((col for col in df.columns if col.lower() == 'sector'), None)
        wo_status_col = next((col for col in df.columns if 'wo status' in col.lower() or col.lower() == 'wo status (billed)'), None)
        billing_status_col = next((col for col in df.columns if 'billing status' in col.lower()), None)
        collection_status_col = next((col for col in df.columns if 'collection status' in col.lower()), None)
        nature_col = next((col for col in df.columns if 'nature' in col.lower()), None)
        type_col = next((col for col in df.columns if 'type of work' in col.lower()), None)

        # Find amount columns
        amount_col = None
        billed_col = None
        collected_col = None
        receivable_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'amount' in col_lower and 'excl' in col_lower and 'gst' in col_lower:
                if 'billed' not in col_lower and 'collected' not in col_lower:
                    amount_col = col
            if 'billed value' in col_lower and 'excl' in col_lower:
                billed_col = col
            if 'collected' in col_lower:
                collected_col = col
            if 'receivable' in col_lower:
                receivable_col = col

        # Financial summaries
        if amount_col:
            numeric_values = pd.to_numeric(df[amount_col], errors='coerce')
            total_order = float(numeric_values.sum())
            summary["total_order_value"] = total_order
            summary["total_order_value_formatted"] = format_indian_currency(total_order)
            summary["orders_with_value"] = int(numeric_values.notna().sum())

        if billed_col:
            billed_values = pd.to_numeric(df[billed_col], errors='coerce')
            total_billed = float(billed_values.sum())
            summary["total_billed_value"] = total_billed
            summary["total_billed_value_formatted"] = format_indian_currency(total_billed)

        if collected_col:
            collected_values = pd.to_numeric(df[collected_col], errors='coerce')
            total_collected = float(collected_values.sum())
            summary["total_collected_value"] = total_collected
            summary["total_collected_value_formatted"] = format_indian_currency(total_collected)

        if receivable_col:
            receivable_values = pd.to_numeric(df[receivable_col], errors='coerce')
            total_receivable = float(receivable_values.sum())
            summary["total_receivable"] = total_receivable
            summary["total_receivable_formatted"] = format_indian_currency(total_receivable)

        # Status breakdowns
        if exec_status_col:
            summary["execution_status_breakdown"] = df[exec_status_col].value_counts().to_dict()

        if wo_status_col:
            summary["wo_status_breakdown"] = df[wo_status_col].value_counts().to_dict()

        if billing_status_col:
            summary["billing_status_breakdown"] = df[billing_status_col].value_counts().to_dict()

        if collection_status_col:
            summary["collection_status_breakdown"] = df[collection_status_col].value_counts().to_dict()

        # Sector breakdown
        if sector_col:
            summary["sector_breakdown"] = df[sector_col].value_counts().to_dict()

        # Nature of work breakdown
        if nature_col:
            summary["nature_of_work_breakdown"] = df[nature_col].value_counts().to_dict()

        # Type of work breakdown
        if type_col:
            summary["type_of_work_breakdown"] = df[type_col].value_counts().to_dict()

        # Cross-tabulated data
        if exec_status_col and sector_col:
            cross_tab = df.groupby([exec_status_col, sector_col]).size().unstack(fill_value=0)
            summary["orders_by_status_and_sector"] = cross_tab.to_dict()

        # Value by sector
        if amount_col and sector_col:
            value_by_sector = df.groupby(sector_col)[amount_col].apply(
                lambda x: pd.to_numeric(x, errors='coerce').sum()
            ).to_dict()
            summary["value_by_sector"] = {k: float(v) for k, v in value_by_sector.items()}
            summary["value_by_sector_formatted"] = format_value_dict(value_by_sector)

        # Value by execution status
        if amount_col and exec_status_col:
            value_by_status = df.groupby(exec_status_col)[amount_col].apply(
                lambda x: pd.to_numeric(x, errors='coerce').sum()
            ).to_dict()
            summary["value_by_execution_status"] = {k: float(v) for k, v in value_by_status.items()}
            summary["value_by_execution_status_formatted"] = format_value_dict(value_by_status)

        # DELAYED WORK ORDERS - Past end date but not completed
        from datetime import datetime
        end_date_col = next((col for col in df.columns if 'probable end' in col.lower()), None)
        if end_date_col and exec_status_col:
            df_copy = df.copy()
            df_copy[end_date_col] = pd.to_datetime(df_copy[end_date_col], errors='coerce')
            today = datetime.now()

            # Delayed = past end date AND not completed
            not_completed = df_copy[~df_copy[exec_status_col].isin(['Completed'])]
            delayed = not_completed[not_completed[end_date_col] < today]

            summary["delayed_work_orders"] = {
                "count": len(delayed),
                "definition": "Work orders past their Probable End Date that are not marked as Completed"
            }

            if sector_col and len(delayed) > 0:
                summary["delayed_work_orders"]["by_sector"] = delayed[sector_col].value_counts().to_dict()

            if exec_status_col and len(delayed) > 0:
                summary["delayed_work_orders"]["by_execution_status"] = delayed[exec_status_col].value_counts().to_dict()

            if amount_col and len(delayed) > 0:
                delayed_value = pd.to_numeric(delayed[amount_col], errors='coerce').sum()
                summary["delayed_work_orders"]["total_value"] = float(delayed_value)
                summary["delayed_work_orders"]["total_value_formatted"] = format_indian_currency(delayed_value)

        # DATA QUALITY INFO
        summary["data_quality"] = {
            "total_records": len(df),
            "missing_values": {}
        }
        for col in df.columns:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                null_pct = round((null_count / len(df)) * 100, 1)
                summary["data_quality"]["missing_values"][col] = {
                    "count": null_count,
                    "percentage": f"{null_pct}%"
                }

        return summary

    def clear_traces(self):
        """Clear tool traces for new query"""
        self.tool_traces = []

    def get_traces(self) -> List[Dict]:
        """Get tool execution traces"""
        return self.tool_traces

    def chat(self, user_message: str) -> Generator[str, None, None]:
        """
        Process user message and generate response with tool calls
        Yields partial responses for streaming
        """
        self.clear_traces()

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Prepare messages for API call
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + self.conversation_history[-10:]  # Keep last 10 messages for context

        try:
            # First call - may include tool calls
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=4096
            )

            assistant_message = response.choices[0].message

            # Check if there are tool calls
            if assistant_message.tool_calls:
                # Process each tool call
                tool_results = []
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                    yield f"🔧 Calling tool: **{tool_name}**\n"

                    result = self._execute_tool(tool_name, tool_args)

                    # Summarize result for the model (avoid sending full data)
                    if "data" in result and len(result.get("data", [])) > 0:
                        result_summary = {
                            "records_count": len(result["data"]),
                            "quality_report": result.get("quality_report", {}),
                            "summary": result.get("summary", {}),
                            "sample_data": result["data"][:5] if result["data"] else []  # Send sample for context
                        }
                    else:
                        result_summary = result

                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": json.dumps(result_summary, default=str)
                    })

                    yield f"✅ Retrieved {len(result.get('data', []))} records\n\n"

                # Add assistant message with tool calls to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })

                # Add tool results to history
                for result in tool_results:
                    self.conversation_history.append(result)

                # Second call - generate final response with tool results
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT}
                ] + self.conversation_history[-15:]

                final_response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096
                )

                final_content = final_response.choices[0].message.content

                # Add final response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_content
                })

                yield final_content

            else:
                # No tool calls - direct response
                content = assistant_message.content or "I couldn't generate a response. Please try again."

                self.conversation_history.append({
                    "role": "assistant",
                    "content": content
                })

                yield content

        except Exception as e:
            error_message = f"Error processing request: {str(e)}"
            yield error_message

    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        self.tool_traces = []
