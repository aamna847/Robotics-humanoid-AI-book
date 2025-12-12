# ActionStep.msg
# A single step in a robot plan

string action_type            # Type of action to perform
                        # Values: "navigate", "perceive", "manipulate", "wait", "speak", "custom"
string parameters             # Action-specific parameters in JSON format
float64 timeout               # Maximum time to wait for this action (seconds) (default: 30.0)