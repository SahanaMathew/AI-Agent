"""
Monday.com API Client
Handles all GraphQL API calls to Monday.com
"""

import requests
import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

class MondayClient:
    """Client for Monday.com GraphQL API"""

    API_URL = "https://api.monday.com/v2"

    def __init__(self):
        self.api_token = os.getenv("MONDAY_API_TOKEN")
        self.deals_board_id = os.getenv("DEALS_BOARD_ID")
        self.work_orders_board_id = os.getenv("WORK_ORDERS_BOARD_ID")

        if not self.api_token:
            raise ValueError("MONDAY_API_TOKEN not found in environment variables")

        self.headers = {
            "Authorization": self.api_token,
            "Content-Type": "application/json"
        }

    def _execute_query(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a GraphQL query against Monday.com API"""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        response = requests.post(self.API_URL, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_boards(self) -> List[Dict]:
        """Get all boards in the workspace"""
        query = """
        {
            boards(limit: 50) {
                id
                name
                items_count
            }
        }
        """
        result = self._execute_query(query)
        return result.get("data", {}).get("boards", [])

    def get_board_columns(self, board_id: str) -> List[Dict]:
        """Get column schema for a specific board"""
        query = """
        query ($boardId: [ID!]) {
            boards(ids: $boardId) {
                columns {
                    id
                    title
                    type
                }
            }
        }
        """
        result = self._execute_query(query, {"boardId": [board_id]})
        boards = result.get("data", {}).get("boards", [])
        if boards:
            return boards[0].get("columns", [])
        return []

    def get_board_items(self, board_id: str, limit: int = 500) -> List[Dict]:
        """Get all items from a board with their column values"""
        query = """
        query ($boardId: [ID!], $limit: Int) {
            boards(ids: $boardId) {
                items_page(limit: $limit) {
                    items {
                        id
                        name
                        column_values {
                            id
                            column {
                                title
                            }
                            text
                            value
                        }
                    }
                }
            }
        }
        """
        result = self._execute_query(query, {"boardId": [board_id], "limit": limit})
        boards = result.get("data", {}).get("boards", [])
        if boards:
            return boards[0].get("items_page", {}).get("items", [])
        return []

    def get_deals_data(self, limit: int = 500) -> List[Dict]:
        """Get all deals from the Deals board"""
        return self.get_board_items(self.deals_board_id, limit)

    def get_work_orders_data(self, limit: int = 500) -> List[Dict]:
        """Get all work orders from the Work Orders board"""
        return self.get_board_items(self.work_orders_board_id, limit)

    def get_deals_columns(self) -> List[Dict]:
        """Get column schema for Deals board"""
        return self.get_board_columns(self.deals_board_id)

    def get_work_orders_columns(self) -> List[Dict]:
        """Get column schema for Work Orders board"""
        return self.get_board_columns(self.work_orders_board_id)


def parse_items_to_records(items: List[Dict]) -> List[Dict]:
    """Convert Monday.com items to flat dictionary records"""
    records = []
    for item in items:
        record = {"Name": item["name"]}
        for col_val in item.get("column_values", []):
            col_title = col_val.get("column", {}).get("title", col_val["id"])
            record[col_title] = col_val.get("text", "")
        records.append(record)
    return records
