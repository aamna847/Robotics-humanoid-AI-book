# PlanStep.msg
# A single step in a robot plan

string id                     # Unique identifier
string step_type              # Type of step to perform
                        # Values: "navigate", "perceive", "manipulate", "wait", "custom"
string parameters             # Step-specific parameters in JSON format
string[] dependencies         # IDs of dependent steps
string[] success_criteria     # Criteria for step completion