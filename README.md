# Automated VAR

Automated video assistant referee; this repository automates some of the functionalities performed by [VAR](https://www.premierleague.com/VAR). The included functionalities include: 

1. Mistaken identity, i.e. the referee awards a yellow or red card to the wrong player. Automated VAR uses object tracking to ensure the correct player is disciplined.
2. Offside calls. Automated VAR checks if a player received the ball in an offside position. 

Examples of both functionalities are provided below.

## Requirements

### Locally
If you are running the application locally, install the dependencies via ```pip install -r requirements.txt```. See the [Mistaken identity](#mistaken-identity) and [Offside calls](#offisde-calls) sections below for running the corresponding functionalities.

### Via Docker
See [Docker](https://www.docker.com/get-started/) for instructions on installing Docker if you don't already have it set up. If/once Docker is set up, follow the proceeding 
instructions for running the container and executing the application: 

1. Pull the image:
2. Run the container: 

## Model Fine-tuning
The [YOLOV5s](https://pypi.org/project/yolov5/) model is used for detection, fine-tuned on the [SoccerNet dataset](https://drive.google.com/drive/folders/17w9yhEDZS7gLdZGjiwPQytLz3-iTUpKm). 

## Mistaken identity

## Offisde calls
