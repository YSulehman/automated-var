import os
import cv2

class IdentityCheck:
    def __init__(self):
        pass

    def run_video_tracker(self, video_file: str, model):
        """
        video_file: path to the video 
        model: model to use for object detection and tracking (YOLOV5s)
        """

        # check if the video file exists
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"File {video_file} not found")
        
        # create video capture object, i.e. for reading the video file
        vid_capture = cv2.VideoCapture(video_file)

        # will I also need an output video, i.e. the annotated video? ADD IN ARGUMENTS
        annotated_video = cv2.VideoWriter()


        # use a while loop to read frames from the video
        while vid_capture.isOpened():
            # check if frame is availble to read and get frame (ret is a boolean; true if frame is avilable)
            ret, frame = vid_capture.read()

            # loop through the frames
            if ret:
                # do something with the frame
                pass
            else:
                break

        # release the video capture object, i.e. this releases hardware and software resources
        vid_capture.release()
        annotated_video.release()

        # close all windows
        cv2.destroyAllWindows()