#### your langgraph code
from graph import build_graph

def main():
    goal = input("Enter your goal: ").strip()
    if not goal:
        print("No goal provided.")
        return

    app = build_graph()
    initial_state = {"goal": goal, "plan": [], "current_step": 0, "results": []}

    print(f"\nGoal: {goal}\n")
    final_state = app.invoke(initial_state)

    plan = final_state["plan"]
    print(f"Plan ({len(plan)} steps):")
    for s in plan:
        print(f"  Step {s['step']}: {s['description']} | tool={s.get('tool')}")
    print()

    print("Results:")
    for r in final_state["results"]:
        print(f"\nStep {r['step']}: {r['description']}")
        print(f"  {r['result']}")

if __name__ == "__main__":
    main()
