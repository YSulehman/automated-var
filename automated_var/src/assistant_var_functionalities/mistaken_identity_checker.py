import os
import cv2

class IdentityCheck:
    class_names = ('player-home', 'player-away', 'goalkeeper-home','goalkeeper-away', 'ball', 'referee', 'linesman', 'staff')
    # only interested in player-home, player-away, goalkeeper-home, goalkeeper-away, ball and referee
    class_indices_to_keep = (0, 1, 2, 3, 4, 5)

    def __init__(self):
        pass

    def _class_colours(self, class_index):
        """
        maps class index to a colour for the bounding box.
        0: player-home (green), 1: player-away (blue), 2: ball (red), 3: referee (yellow)
        """
        bounding_box_colours = {'0': (0, 255, 0), '1': (0, 0, 255), '2': (255, 0, 0), '3': (255, 255, 0), '4': (0, 255, 255), '5': (255, 0, 255), 
                                    '6': (255, 255, 255), '7': (0, 0, 0)}

        return bounding_box_colours[str(class_index)]

    def run_video_tracker(self, source_video: str, destination_video: str, model):
        """
        source_video: path to the input video
        destination_video: path to the output video, i.e. the annotated video 
        model: model to use for object detection and tracking (YOLOV5s)
        """

        # check if the video file exists
        if not os.path.exists(source_video):
            raise FileNotFoundError(f"File {source_video} not found")
        
        # create video capture object, i.e. for reading the video file
        vid_capture = cv2.VideoCapture(source_video)

        # we need the width and height of the video to create the output video
        width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # will I also need an output video, i.e. the annotated video? ADD IN ARGUMENTS
        # arguments are the destination video, the codec, the frames per second and the size of the video
        # fourcc: 4-byte code used to specify the video codec, i.e. the video compression format
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        annotated_video = cv2.VideoWriter(destination_video, fourcc, 20, (width, height))


        # use a while loop to read frames from the video
        while vid_capture.isOpened():
            # check if frame is availble to read and get frame (ret is a boolean; true if frame is available)
            ret, frame = vid_capture.read()

            # loop through the frames
            if ret:
                # each frame needs to resized 
                resized_frame = cv2.resize(frame, (width, height))

                # get the bounding boxes (tensor) of the objects in the frame, using the model
                # results is of the form (batch_size (=1 in this case), number of predictions, 6)
                results = model(resized_frame, conf=0.5)
                
                # loop through the results
                for result in results:
                    # result is of the form (number of predictions, 6)
                    # parse first for predictions
                    predictions = result[0]

                    # get the class index, bounding box coordinates and label of the object
                    class_indices = predictions[:, 5]
                    predicted_classes = self.class_names[class_indices]
                    x1, y1, x2, y2 = predictions[:, :4]

                    # get the colour for the bounding box
                    colour = self._class_colours(class_indices)

                    # now, interested in only the player-home, player-away, goalkeeper-home, goalkeeper-away, ball and referee, so filter for these 
                    for index, cls in enumerate(class_indices):
                        if cls in self.class_indices_to_keep:
                            # draw the bounding box on the frame
                            cv2.rectangle(resized_frame, (x1[index], y1[index]), (x2[index], y2[index]), colour, 2)
                            # label the text, args specify the frame, text, position, font, font scale, colour and thickness
                            cv2.putText(resized_frame, predicted_classes, (x1[index] - 10, y1[index] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

                # write the annotated frame to the annotated video
                annotated_video.write(resized_frame)

            # no frame available, i.e. end of video. Break out of the loop    
            else:
                break

        # release the video capture object, i.e. this releases hardware and software resources
        vid_capture.release()
        annotated_video.release()

        # close all windows
        cv2.destroyAllWindows()