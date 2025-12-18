from data_collection import get_b_data
from data_prep import prep_b_data



def refresh():
    print("\n"*50)

def print_title(word=None):
    if not word:
        print(f"\n{'='*60}")
        print("               Welcome to MLB Season Predictor")
        print(f"\n{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"               {word}")
        print(f"\n{'='*60}")

stat_dict = {
    "HR": "Home Runs (HR)",
    "AVG": "Batting Average (AVG)",
    "OPS": "On-base PLus Slugging (OPS)",
    "wRC+": "Weighted Runs Created Plus (wRC+)",
}

refresh()
print_title("Choose output stat")
print("Enter what stat you would like to predict:")
for stat in stat_dict:
    print(f"{stat}: {stat_dict.get(stat)}")

num_choices = 4

while True:
    choice = input(">>Enter choice (HR, AVG, etc.): ")
    break

    #TODO: error check


refresh()
print_title()
print(f"Chosen stat = {stat_dict.get(choice)}")
print("\n")

#get raw database from pybaseball
get_b_data.run()
prep_b_data.run(choice)

refresh()
print_title()

