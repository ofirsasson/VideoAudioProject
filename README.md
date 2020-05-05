
VideoAudio was made as project exmaple.

Name of the project:
VideoAudio

Description:
VideoAudio is a project that focuses on combining different types of models to create a single user profile.
In the project I used both voice and face identification models with an input from a deepspeech (tensorflow implemented) model.

VideoAudio can see who is talking hear what he's saying and put it on the screen. 

Installation:
Requires installation of DeepSpeech, OpenCV and other packages.
For further iformation on packages see requirements.txt

Usage:
There are two main proccesses happenning in the software:
1. Update: Creating a new user profile by making its GMM model and adding user's face for the face identification model. -> updating-script.sh for complete stages
2. Activation: Recognizing active profiles and transcribing what they say on screen. -> activation-script.sh for complete stages

purpose of the main python scripts:
activation-script.py: Identifies profiles on the camera, gets the .wav files at Data/tmp and identifies who they belong to and if the profile who it belongs to still on camera. If so, the script will show the words that were recognized.

mic_vad_streaming.py: A DeepSpeech speech recognizer used to recognize and save any sound input from profiles in both proccesses.

modeltraining.py: Trains a GMM model for each new profile created.

faces-train.py: Trains a face identification model based on the pictures of the new profile.


License:
MIT License

Copyright (c) [2020] [Ofir Sasson]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
# VideoAudioProject
# VideoAudioProject
