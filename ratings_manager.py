import json
import os
from datetime import datetime

# ✅ Use absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.json")

def save_rating(rating, feedback=None):
    """
    Save only rating and feedback - simplified version.
    
    Args:
        rating: int (1-5)
        feedback: str or None
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    entry = {
        "rating": rating,
        "feedback": feedback or "",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        # Read existing ratings
        if os.path.exists(RATINGS_FILE):
            with open(RATINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        data.append(entry)

        # Save to file
        with open(RATINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print("✅ Rating saved:", entry)
        return entry

    except Exception as e:
        print("❌ Failed to save rating:", e)
        raise

def get_all_ratings():
    """Get all ratings."""
    try:
        if os.path.exists(RATINGS_FILE):
            with open(RATINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading ratings: {e}")
        return []

def get_rating_stats():
    """Get rating statistics."""
    ratings = get_all_ratings()
    
    if not ratings:
        return {
            "total_ratings": 0,
            "average_rating": 0.0,
            "rating_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "with_feedback": 0
        }
    
    rating_values = [r["rating"] for r in ratings]
    with_feedback = sum(1 for r in ratings if r.get("feedback"))
    
    return {
        "total_ratings": len(ratings),
        "average_rating": round(sum(rating_values) / len(rating_values), 2),
        "rating_distribution": {i: rating_values.count(i) for i in range(1, 6)},
        "with_feedback": with_feedback,
        "latest_ratings": ratings[-10:][::-1]
    }

# Test
if __name__ == "__main__":
    print("Testing simplified ratings...")
    save_rating(5, "Great bot!")
    save_rating(4, None)
    print("\nStats:", get_rating_stats())