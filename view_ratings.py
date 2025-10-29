from ratings_manager import get_all_ratings, get_rating_stats, get_low_rated_responses
from datetime import datetime

def print_separator(char="=", length=80):
    print(char * length)

def display_rating_stars(rating):
    return "⭐" * rating + "☆" * (5 - rating)

def view_all_ratings():
    """Display all ratings - FIXED for simplified format."""
    print_separator()
    print("📊 ALL RATINGS")
    print_separator()
    
    ratings = get_all_ratings()
    
    if not ratings:
        print("No ratings found yet.")
        return
    
    for i, rating in enumerate(ratings, 1):
        print(f"\n#{i} - {rating['timestamp']}")
        print(f"Rating: {display_rating_stars(rating['rating'])} ({rating['rating']}/5)")
        
        if rating.get('feedback') and rating['feedback'].strip():
            print(f"\n💭 User Feedback:")
            print(f"   {rating['feedback']}")
        else:
            print("\n💭 No feedback provided")
        
        print_separator("-")

def view_statistics():
    """Display rating statistics."""
    print_separator()
    print("📈 RATING STATISTICS")
    print_separator()
    
    stats = get_rating_stats()
    
    print(f"\nTotal Ratings: {stats['total_ratings']}")
    print(f"Average Rating: {stats['average_rating']}/5.0 {display_rating_stars(int(stats['average_rating']))}")
    print(f"Ratings with Feedback: {stats['with_feedback']}")
    
    print("\n📊 Rating Distribution:")
    for stars in range(5, 0, -1):
        count = stats['rating_distribution'][stars]
        bar = "█" * count
        print(f"  {stars} ⭐: {bar} ({count})")

def view_low_ratings(threshold=2):
    """Display low ratings."""
    print_separator()
    print(f"⚠️ LOW RATED RESPONSES (≤{threshold} stars)")
    print_separator()
    
    low_ratings = get_low_rated_responses(threshold)
    
    if not low_ratings:
        print(f"\n✅ Great! No ratings below {threshold} stars.")
        return
    
    print(f"\nFound {len(low_ratings)} low ratings:\n")
    
    for i, rating in enumerate(low_ratings, 1):
        print(f"#{i} - {display_rating_stars(rating['rating'])} ({rating['rating']}/5)")
        print(f"Time: {rating['timestamp']}")
        
        if rating.get('feedback'):
            print(f"Feedback: {rating['feedback']}")
        
        print_separator("-")

def main_menu():
    """Interactive menu."""
    while True:
        print("\n")
        print_separator("=")
        print("📊 HR CHATBOT RATING VIEWER")
        print_separator("=")
        print("\n1. View All Ratings")
        print("2. View Statistics")
        print("3. View Low Rated Responses (≤2 stars)")
        print("4. View Low Rated Responses (≤3 stars)")
        print("5. Exit")
        print()
        
        choice = input("Select an option (1-5): ").strip()
        
        if choice == "1":
            view_all_ratings()
        elif choice == "2":
            view_statistics()
        elif choice == "3":
            view_low_ratings(threshold=2)
        elif choice == "4":
            view_low_ratings(threshold=3)
        elif choice == "5":
            print("\n👋 Goodbye!\n")
            break
        else:
            print("❌ Invalid option.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")