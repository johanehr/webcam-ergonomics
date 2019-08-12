# webcam-ergonomics
This is a computer vision project with the goal of estimating head position relative to the webcam, based on the assumption that the user mostly looks straight ahead. The constant distance between the eyes then allows for an estimate of distance from the webcam, as well as a general angle. If the head position strays outside a "safe-zone" around a neutral, ergonomically desired position for too long, a warning sound will be played, prompting the user to correct their posture. This neutral position and other settings can be adjusted by editing the settings.json file.

Note that the focal length ("f" in settings.json) can be roughly estimated as follows:

f = cot(a/2)w/2

f: focal length
a: horizontal field of view
w: horizontal resolution (printed once in terminal)

The eye detection used does not seem to work very well for slim eyes - the author included. This should be fixed for use outside a simple proof of concept.
