# webcam-ergonomics
This is a computer vision project with the goal of estimating head position relative to the webcam, based on the assumption that the user mostly looks straight ahead. The constant distance between the eyes then allows for an estimate of the head's position relative to the webcam, assuming a projective camera model.

If the head strays outside a "safe-zone" around a neutral, ergonomically desired position for too long, a warning sound will be played, prompting the user to correct their posture. This neutral position and other settings can be adjusted by editing the "settings.json" file. For a visual representation and position estimate, use the flag "-L" when running the script to include the live view.

The focal length ("f" in settings.json) can be roughly estimated as follows:

f = cot(a/2)w/2

f: focal length
a: horizontal field of view
w: horizontal resolution (printed once in terminal)

The eye detection used does not seem to work very well for slim eyes - the author included. This should be fixed for use outside a simple proof of concept, for example by detecting the face and eyes using a modern neural network technique, instead of Haar Cascade classifiers. Note that the face is assumed to be directed roughly towards the webcam's image plane. 
