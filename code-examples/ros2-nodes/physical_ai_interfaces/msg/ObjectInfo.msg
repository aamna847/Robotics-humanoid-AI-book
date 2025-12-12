# ObjectInfo.msg
# Information about an object in the environment

string name                    # Name of the object
geometry_msgs/Pose pose       # Pose of the object in space
geometry_msgs/Vector3 dimensions # Dimensions of the object (length, width, height)
float64 mass                  # Mass of the object in kg (default: 0.5)
string material               # Material of the object (default: "plastic")
                        # Values: "plastic", "metal", "wood", "fabric", "glass", "other"