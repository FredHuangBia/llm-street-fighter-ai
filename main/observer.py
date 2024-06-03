import numpy as np

#KEN_RED = [248, 0, 0]
#KEN_GREEN = [88, 176, 40]

X_SIZE = 128
Y_SIZE = 100

class Observer():
    def __init__(self, character_color, ennemy_color):
        self.ennemy_color = ennemy_color
        self.character_color = character_color
        self.observations = []
        return

    def detect_position_from_color(
        self, observation: dict, color: list, epsilon=1, save_frame: bool = False
    ) -> tuple:
        """
        Convert the observation from pixels to player coordinates.
        It works by finding the first pixel that matches the color.
        Returns a tuple of (x, y) coordinates.
        - x is between 0 and 384
        - y is between 0 and 224
        """
        frame = observation["frame"]
        # the screen is a np.array of RGB colors (3 channels)
        # Select the frames where the characters play: between 80 vertical and 200 vertical

        # dump the observation to a file for debugging
        if save_frame:
            np.save("observation.npy", frame)

        frame = frame[50:90, :]

        # Detect the red color of Ken
        diff = np.linalg.norm(np.array(frame) - np.array(color), axis=2)
        mask = diff < epsilon

        # Return the index where the red color is detected
        coordinates = mask.nonzero()

        if len(coordinates[0]) == 0:
            return None

        # Add back the vertical offset
        first_match = (coordinates[1][0], coordinates[0][0] + 50)

        return first_match

    def observe(self, observation: dict):
        """
        The robot will observe the environment by calling this method.
        The latest observations are at the end of the list.
        """

        ### debug ##########
        #print(f" robot Observation: {observation}")
        ######################
        # detect the position of characters and ennemy based on color
        observation["character_position"] = self.detect_position_from_color(
            observation, self.character_color
        )
        observation["ennemy_position"] = self.detect_position_from_color(
            observation, self.ennemy_color
        )

        self.observations.append(observation)
        # we delete the oldest observation if we have more than 10 observations
        if len(self.observations) > 10:
            self.observations.pop(0)


        # Keep track of the current direction by checking the position of the character
        # and the ennemy
        character_position = observation.get("character_position")
        ennemy_position = observation.get("ennemy_position")
        if (
            character_position is not None
            and ennemy_position is not None
            and len(character_position) == 2
            and len(ennemy_position) == 2
        ):
            if character_position[0] < ennemy_position[0]:
                self.current_direction = "Right"
            else:
                self.current_direction = "Left"

    def context_prompt(self) -> str:
        """
        Return a str of the context
        "The observation for you is Left"
        "The observation for the opponent is Left+Up"
        "The action history is Up"
        """

        # Create the position prompt
        obs_own = self.observations[-1]["character_position"]
        obs_opp = self.observations[-1]["ennemy_position"]

        if obs_own is not None and obs_opp is not None:
            relative_position = np.array(obs_own) - np.array(obs_opp)
            normalized_relative_position = [
                relative_position[0] / X_SIZE,
                relative_position[1] / Y_SIZE,
            ]
        else:
            normalized_relative_position = [0.3, 0]

        position_prompt = ""
        if abs(normalized_relative_position[0]) > 0.1:
            position_prompt += (
                "You are very far from the opponent. Move closer to the opponent. "
            )
            if normalized_relative_position[0] < 0:
                position_prompt += "Your opponent is on the right."
            else:
                position_prompt += "Your opponent is on the left."

        else:
            position_prompt += "You are close to the opponent. You should attack him."

        # Assemble everything
        context = f"""{position_prompt}"""

        return context