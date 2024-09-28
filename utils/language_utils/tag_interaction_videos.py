import cv2
import glob
import numpy as np
import json 
import os 

import tkinter as tk
from tkinter import simpledialog

# 6 types 
type_dict = {
    "whitechair": "chair",
    "woodchair": "chair", 
    "white": "chair", # For the sequences with inconsistet names. 
    "smalltable": "table",
    "largetable": "table",
    "bugtable": "table", # For the sequences with typo. 
    "smallbox": "box",
    "largebox": "box",
    "plasticbox": "box", 
    "suitcase": "box", 
    "trashcan": "box", 
    "monitor": "monitor",
    "floorlamp": "floorlamp",
    "tripod": "tripod",
    "clothesstand": "tripod",
}

# Each type has a couple of descriptions to selecte from. 
text_dict = {
    "chair": {
        "1": "Facing the back of the chair, lift the chair and then place the chair onto the floor.",
        "2": "Lift and move the chair.",
        "3": "Grab the top of the chair, swing the chair.",
        "4": "Lift the chair over your head, walk and place the chair onto the floor.",
        "5": "Put your hand on the back of the chair at the top. Pull on it to move it across the floor.",
        "6": "Lift the chair, rotate the chair and set it back down.",
        "7": "Use the foot to scoot the chair to change its orientation.",
        "8": "Push the chair, then turn yourself around so you can then drag the chair behind you.",
        "9": "Hold the chair and turn it around to face a diffferent orientation.",
        "a": "Grab one of the chair's legs and tilt it at an angle.",
        "s": "Kick the chair across the room.",
        "d": "Lift the chair, flip it upside down and place it on top of the table.",
        "f": "Move the chair upside down from the table to the floor.",
        "g": "Lift the chair, flip it upside down and place it on top of the table. And then move the chair upside down from the table to the floor.",
        "h": "Lift and move the chair. Then kick the chair to move.", # Additional one. 
    }, 
    "table": {
        "1": "Pull the table to the desired location.",
        "2": "Lift and move the table to the desired location.",
        "3": "Lift the table above your head, spin it and put the table down.",
        "4": "Lift the table above your head, walk, and put the table down.",
        "5": "Lift the table by one of the edges, rotate it, drag the table and place the legs back down.",
        "6": "Push to move the table.",
        "7": "Kick the table to move across the room.",
        "8": "Push the table and then drag it to move, then set the table down.",
        "9": "Lift the table, so only two legs are off the floor. Slide your feet and rotate the table as you slide. Lower the table with your hands.",
    }, 
    "box": {
        "1": "Push and move the box to the desired location.",
        "2": "Lift the box, carry it a few steps and place it on top of a table.",
        "3": "Pick up the box from the table and put down the box onto the floor.", # Additional one. 
        "4": "Lift the box, carry it toward the desired location, and put it down onto the floor.",
        "5": "Lift, rotate, and place the box back down.",
        "6": "Kick the box to slide, lift the box and place it on top of a nearby shelf.",
        "7": "Kick the box, then lift the box and move it to the desired location of the floor.", # Additional one. For suitcase. 
        "8": "Kick the box to move the box across the room.",
        "9": "Pull the box toward you and release the box.",
        "a": "Push the box and then pull the the box towards you, causing the box to slide across the floor.",
    }, 
    "monitor": {
        "1": "Pick up the monitor, walk towards a desk, and put the monitor on the desk.",
        "2": "Pick up the monitor from the desk and put down the monitor onto the floor.", # Additional one. 
        "3": "Pick up the monitor, put the monitor on the desk. Then pick up the monitor from the desk and put down the monitor onto the floor.", # Additional one. 
        "4": "Pick up the monitor and move the monitor to the desired location of the floor.", # Additional one. 
        "5": "Lift the monitor, place the monitor on the table, then rotate the monitor to adjust its orientation.",
        "6": "Lift the monitor, rotate the monitor and put down the monitor.",
        "7": "Grasp the sides of the monitor, tilt it and slide it across a surface.", 
        "8": "Lift the monitor up, walk backward, and place the monitor on the ground.",
        "9": "Pick up the monitor and then rotate the monitor while putting it down.",
    }, 
    "floorlamp": {
        "1": "Lift and move the floorlamp to the desired location.",
        "2": "Pull and move the floorlamp to the desired location.",
        "3": "Kick the base of the floorlamp to move.",
        "4": "Lift and adjust the floorlamp to a different orientation.",
    }, 
    "tripod": {
        "1": "Pick up and move the tripod to the desired location.",
        "2": "Pull and move the tripod to the desired location.",
        "3": "Put the tripod down onto the floor, then pick up the tripod.",
        "4": "Put the tripod down onto the floor.",
        "5": "Pick up the fallen tripod from the floor.",
        "6": "Hold and turn the tripod around to a different orientation.",
        "7": "Kick the tripod to move.",
        # "7": "Pick up, rotate, and put down the tripod.", # A special type of picking up and move. 
    }, 
}

def tag_video(video_tags, video_file, tag):
    video_tags[video_file] = tag

if __name__ == "__main__":
    # Please modify video_directory and dest_res_folder accordingly. 

    # Path to your videos directory
    video_directory = '/Users/jiamanli/Desktop/Research/2023_Summer_Meta_Intern/for_language_anno/sub1/rendered_mp4s'

    dest_res_folder = '/Users/jiamanli/Desktop/omomo_data_text_anno_res'
    if not os.path.exists(dest_res_folder):
        os.makedirs(dest_res_folder) 

    # Get list of all video files in the directory
    video_files = glob.glob(f"{video_directory}/*.mp4")

    video_files.sort() 

    # Create a Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    for video_file in video_files:
        video_name = video_file.split("/")[-1]

        # Save video tags to .json files. 
        curr_json_path = os.path.join(dest_res_folder, video_name.replace(".mp4", ".json"))

        # If this video is already annotated, then skip to the next video for annotation. 
        if os.path.exists(curr_json_path):
            continue 

        object_name = video_file.split("/")[-1].split("_")[1]
        if "yifengsuitcase" in video_file:
            object_name = "suitcase" # For naming isssues of some sequences. 

        if object_name not in type_dict:
            continue 
        type_name = type_dict[object_name] 

        tags = text_dict[type_name] 

        video_tags = {} 

        # Open video file
        cap = cv2.VideoCapture(video_file)

        while(cap.isOpened()):
            # Read frame
            ret, frame = cap.read()

            # If frame is read correctly, ret is True
            if not ret:
                print("Video ended. Waiting for tag input...")
                break

            # Create an empty image for tag descriptions with the same height as the frame
            # The color is set to white
            desc_img = np.ones((frame.shape[0], 1000, 3), np.uint8) * 255

            # Show the current tags on the separate image
            y = 20  # initial y location for tags
            cv2.putText(desc_img, "Press q to quit, press n to skip current seq.", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            y += 40 
            cv2.putText(desc_img, "Tag Descriptions", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            y += 25  # adjust for next tag line
            for k, v in tags.items():
                cv2.putText(desc_img, "Press {} for: {}".format(k, v), (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                y += 25  # adjust for next tag line

            # Concatenate frame and descriptions
            combined = np.hstack((frame, desc_img))

            # Show combined frame and descriptions
            cv2.imshow('Frame and Tag Descriptions', combined)

            # Wait for a key press for next frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # Break the loop if 'q' is pressed
                break

        # Now video ended, we wait for a tag
        quit_anno = False 
        skip_seq = False 
        while True:
            key = cv2.waitKey(0)  # delay of 0 means "wait indefinitely"

            if key != -1:
                # Convert to character
                key = chr(key & 0xFF)

                if key in tags:
                    # Save tag for current video when a valid tag key is pressed
                    tag_video(video_tags, video_name, tags[key])
                    print(f"Video {video_file} tagged as {tags[key]}")
                    break
                elif key == 't':
                    # If 't' is pressed, open a dialog to enter a new tag
                    answer = simpledialog.askstring("Input", "Enter your tag:", parent=root)
                    if answer is not None:
                        # Save custom tag for current video
                        tag_video(video_tags, video_file, answer)
                        print(f"Video {video_file} tagged as {answer}")
                        break
                elif key == "q":
                    quit_anno = True 
                    break 
                elif key == "n": # Skip this sequence. 
                    skip_seq = True 
                    break 

        cap.release()
        cv2.destroyAllWindows()

        if quit_anno:
            break 

        if skip_seq:
            continue 

        # print("Video tags:", video_tags)

        # Save video tags to .json files. 
        curr_json_path = os.path.join(dest_res_folder, video_name.replace(".mp4", ".json"))
        json.dump(video_tags, open(curr_json_path, 'w'))  
