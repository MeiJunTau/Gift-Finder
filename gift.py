import os
import json
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# 1. Load your .env file
load_dotenv()

# 2. Configure Gemini with Search Grounding
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash", # Use the stable 2.5 production backbone
    api_key=os.getenv("GEMINI_API_KEY"),
    google_search=True
)

def run_holiday_gift_scout():
    # Load your full Family Database
    with open('family.json', 'r') as f:
        family = json.load(f)

    print(f"🚀 Starting Gift Scout for {len(family)} family members...")

    for person in family:
        print(f"\n--- 🔎 Researching for: {person['name']} ---")
        
        # Agent: Uses the person's interests from the JSON
        scout = Agent(
            role="Holiday Shopping Expert",
            goal=f"Find three perfect 2026 holiday gifts for {person['name']}",
            backstory=f"You are a shopping assistant. {person['name']} loves {person['interests']}.",
            llm=gemini_llm,
            verbose=True
        )

        # Task: Specifically tailored to the individual
        hunt_task = Task(
            description=(
                f"Find 3 gift ideas for {person['name']}. "
                f"Interests: {person['interests']}. "
                "Focus on reasonably priced items aournd £50 available for the 2026 season from UK retailers."
            ),
            expected_output="A list with item name, price, and why it fits.",
            agent=scout,
            output_file=f"reports/{person['name']}_gift_guide.md"
        )

        # Run the crew for this specific person
        crew = Crew(agents=[scout], tasks=[hunt_task])
        result = crew.kickoff()
        
        # Save or Print results
        print(f"\n✅ {person['name']}'s Suggestions:\n{result}")

if __name__ == "__main__":
    run_holiday_gift_scout()