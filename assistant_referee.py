import yolov5
import argparse
from automated_var.src.assistant_var_functionalities import mistaken_identity_checker as mic
from automated_var.src.assistant_var_functionalities import offside_checker as oc


# class AssistantReferee(mic.IdentityCheck, oc.OffsideCheck):
#     def __init__(self):
#         pass

def main(args):
    # load the YOLOV5s model
    model = yolov5.load_model("yolov5s")

    if args.command == "identity-check":
        var = mic.IdentityCheck()

        # run video tracker for identity check
        var.run_video_tracker(args.source_video, args.destination_video, model)

    else:
        var = oc.OffsideCheck()

        # run offside check
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-com", "--command", type=str, choices=["identity-check", "offside-check"], required=True, help="which VAR functionality to run")
    parser.add_argument("-sv", "--source-video", type=str, required=True, help="path to the input video")
    parser.add_argument("-dv", "--destination-video", type=str, help="path to the output video")

    args = parser.parse_args()

    main(args)