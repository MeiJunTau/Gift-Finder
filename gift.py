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
        
        # AGENT 1: Uses the person's interests from the JSON
        scout = Agent(
            role="Holiday Shopping Expert",
            goal=f"Find three perfect 2026 holiday gifts for {person['name']}",
            backstory=f"You are a shopping assistant. {person['name']} loves {person['interests']}.",
            llm=gemini_llm,
            verbose=True
        )

        # AGENT 2: The Concierge (The "Stylist")
        concierge = Agent(
            role="Luxury Gift Concierge",
            goal="Transform raw gift ideas into a beautifully formatted, warm, and professional report.",
            backstory=("You are a high-end personal concierge. You don't just list items; "
                        "you tell a story of why each gift is perfect for the recipient. "
                        "Your tone is helpful, elegant, and festive."
            ),
            llm=gemini_llm,
            verbose=True
        )

        # TASK 1: Researching
        hunt_task = Task(
            description=(
                f"Find 3 gift ideas for {person['name']}. "
                f"Interests: {person['interests']}. "
                "Focus on reasonably priced items aournd £50 available for the 2026 season from UK retailers."
            ),
            expected_output="A raw list of 3 gifts with prices and availability.",
            agent=scout
        )
        
        # TASK 2: Formatting (This uses 'context' to link to Task 1)
        style_task = Task(
            description=(
                f"Take the gift ideas found for {person['name']} and format them into a 'Personalized Holiday Gift Guide'. "
                "Use Markdown headers, bullet points, and add a 'Concierge Note' for each item explaining why it fits their specific personality."
            ),
            expected_output="A luxury-formatted Markdown report ready for a high-end client.",
            agent=concierge,
            context=[hunt_task], 
            output_file=f"reports/{person['name']}_gift_guide.md"
        )

        # RUN THE CREW
        crew = Crew(
            agents=[scout, concierge],
            tasks=[hunt_task, style_task],
            process="sequential", # Ensures Scout finishes before Concierge starts
            memory=False
        )

        result = crew.kickoff()
        
        # Save or Print results
        print(f"\n✅ {person['name']}'s Suggestions:\n{result}")

if __name__ == "__main__":
    run_holiday_gift_scout()
