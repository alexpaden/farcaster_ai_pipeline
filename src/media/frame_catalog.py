import os
import asyncio
import requests
import json
from dotenv import load_dotenv
import logging
from typing import Optional, Union, List, Dict, Any

# Adjust import path based on your project structure if db.py is in a different relative location
# Assuming db.py is in src/db/connect.py and this file is in src/media/frame_catalog.py
from ..db.connect import db, setup_logging

# Load environment variables from .env file
load_dotenv()

# Setup logging
logger = setup_logging()

NEYNAR_API_KEY = os.getenv("NEYNAR_API_KEY")
NEYNAR_API_URL = "https://api.neynar.com/v2/farcaster/frame/catalog"
TABLE_NAME = "frame_catalog" # Will be created in 'unbias' schema due to search_path

def get_longest_str(*args: str) -> Optional[str]:
    """Returns the longest string among the arguments, or None if all are None/empty."""
    longest_s = ""
    for s in args:
        if s and isinstance(s, str) and len(s) > len(longest_s):
            longest_s = s
    return longest_s if longest_s else None

async def create_frame_catalog_table():
    """Creates the frame_catalog table if it doesn't exist."""
    async with db.pool.acquire() as conn:
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS unbias.{TABLE_NAME} (
                frames_url TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                tags TEXT[],
                raw_data JSONB,
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logger.info(f"Table '{TABLE_NAME}' ensured to exist.")

def extract_frame_data(frame: dict) -> Optional[Dict[str, Any]]:
    """Extracts and transforms relevant data from a single frame object."""
    if not frame or not isinstance(frame, dict):
        return None

    frames_url = frame.get("frames_url")
    if not frames_url:
        logger.warning(f"Frame missing frames_url: {frame.get('title') or 'Unknown title'}")
        return None

    # Safely access nested dictionaries
    manifest = frame.get("manifest", {})
    manifest_frame = manifest.get("frame", {}) if isinstance(manifest, dict) else {}
    metadata = frame.get("metadata", {})
    metadata_html = metadata.get("html", {}) if isinstance(metadata, dict) else {}

    # Titles
    title_candidates = [
        manifest_frame.get("name"),
        manifest_frame.get("title"), # As per user request, though not in typical manifest structure
        manifest_frame.get("og_title"),
        metadata_html.get("ogTitle"),
        frame.get("title")
    ]
    selected_title = get_longest_str(*(c for c in title_candidates if c))

    # Descriptions
    description_candidates = [
        manifest_frame.get("description"),
        manifest_frame.get("og_description"),
        metadata_html.get("ogDescription")
    ]
    selected_description = get_longest_str(*(c for c in description_candidates if c))
    
    # Tags
    tags_list = manifest_frame.get("tags")
    if not isinstance(tags_list, list) or not any(tags_list): # Check if it's a non-empty list
        primary_category = manifest_frame.get("primary_category")
        current_tags = [primary_category] if primary_category and isinstance(primary_category, str) else []
    else:
        current_tags = [str(tag) for tag in tags_list if tag and isinstance(tag, (str, int, float))]
    
    # Ensure tags is an empty list if no tags found, for TEXT[] compatibility
    current_tags = current_tags if current_tags else []


    return {
        "frames_url": frames_url,
        "title": selected_title,
        "description": selected_description,
        "tags": current_tags,
        "raw_data": frame # Store the whole original frame object as JSONB
    }

async def fetch_all_frames_from_neynar() -> List[Dict[str, Any]]:
    """Fetches all frames from the Neynar API, handling pagination."""
    if not NEYNAR_API_KEY:
        logger.error("NEYNAR_API_KEY not found in environment variables.")
        return []

    all_frames_data = []
    cursor = None
    headers = {"api_key": NEYNAR_API_KEY, "accept": "application/json"}
    
    page_count = 0
    while True:
        page_count += 1
        params = {"limit": 100, "time_window": "7d"} # As per user spec
        if cursor:
            params["cursor"] = cursor

        logger.info(f"Fetching page {page_count} from Neynar Frame Catalog API (cursor: {cursor})...")
        try:
            response = await asyncio.to_thread(
                requests.get, NEYNAR_API_URL, headers=headers, params=params, timeout=30
            )
            response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
            data = response.json()
            
            frames_in_response = data.get("frames", [])
            if not frames_in_response:
                logger.info("No more frames found in the response.")
                break

            for frame_json in frames_in_response:
                processed_frame = extract_frame_data(frame_json)
                if processed_frame:
                    all_frames_data.append(processed_frame)
            
            next_cursor = data.get("next", {}).get("cursor")
            if not next_cursor:
                logger.info("No next cursor found. Reached the end of pagination.")
                break
            cursor = next_cursor
            # Optional: add a small delay to be polite to the API
            # await asyncio.sleep(0.2) 

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching frames: {e}. Response: {e.response.text}")
            break
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching frames: {e}")
            break
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}. Response text: {response.text}")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred during API fetch: {e}")
            break
            
    logger.info(f"Fetched a total of {len(all_frames_data)} frame entries.")
    return all_frames_data

async def upsert_frames_to_db(frames_data: List[Dict[str, Any]]):
    """Upserts a list of processed frame data into the database."""
    if not frames_data:
        logger.info("No frame data to upsert.")
        return

    upserted_count = 0
    failed_count = 0
    async with db.pool.acquire() as conn:
        for frame_item in frames_data:
            try:
                await conn.execute(f"""
                    INSERT INTO unbias.{TABLE_NAME} (frames_url, title, description, tags, raw_data, last_updated)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                    ON CONFLICT (frames_url) DO UPDATE SET
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        tags = EXCLUDED.tags,
                        raw_data = EXCLUDED.raw_data,
                        last_updated = NOW();
                """, frame_item["frames_url"], frame_item["title"], frame_item["description"], 
                     frame_item["tags"], json.dumps(frame_item["raw_data"])) # raw_data needs to be json string for asyncpg
                upserted_count +=1
            except Exception as e:
                logger.error(f"Error upserting frame {frame_item.get('frames_url')}: {e}")
                failed_count +=1
    logger.info(f"Upserted {upserted_count} frames. Failed to upsert {failed_count} frames.")

async def main():
    logger.info("Starting Farcaster Frame Catalog update process...")
    
    if not NEYNAR_API_KEY:
        logger.critical("NEYNAR_API_KEY environment variable is not set. Exiting.")
        return

    await db.initialize_pool(min_size=1, max_size=2) # Initialize the pool from connect.py
    
    try:
        await create_frame_catalog_table()
        
        fetched_frames = await fetch_all_frames_from_neynar()
        
        if fetched_frames:
            await upsert_frames_to_db(fetched_frames)
        else:
            logger.info("No new frames fetched to update in the database.")
            
    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}", exc_info=True)
    finally:
        await db.close_pool() # Close the pool from connect.py
        logger.info("Farcaster Frame Catalog update process finished.")

if __name__ == "__main__":
    asyncio.run(main())
