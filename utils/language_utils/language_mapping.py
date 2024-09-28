import os 
import numpy as np 
import json 

chair_mapping_dict = {
    "Facing the back of the chair, lift the chair and then place the chair onto the floor.": "Facing the back of the chair, lift the chair, move the chair, and then place the chair on the floor.",  
    "Lift and move the chair.": "Lift the chair, move the chair, and put down the chair.", 
    "Grab the top of the chair, swing the chair.": "Grab the top of the chair, swing the chair, and put down the chair.",
    "Lift the chair over your head, walk and place the chair onto the floor.": "Lift the chair over your head, walk and then place the chair on the floor.", 
    "Put your hand on the back of the chair at the top. Pull on it to move it across the floor.": "Put your hand on the back of the chair, pull the chair, and set it back down.", 
    "Lift the chair, rotate the chair and set it back down.": "Lift the chair, rotate the chair, and set it back down.", 
    "Use the foot to scoot the chair to change its orientation.": "Use the foot to scoot the chair to change its orientation.", 
    "Push the chair, then turn yourself around so you can then drag the chair behind you.": "Push the chair, release the hands, then drag the chair, and set it back down.", 
    "Hold the chair and turn it around to face a diffferent orientation.": "Hold the chair and turn it around to face a diffferent orientation.", 
    "Grab one of the chair's legs and tilt it at an angle.": "Grab the chair's legs, tilt the chair.", 
    "Kick the chair across the room.": "Kick the chair, and set it back down.", 
    "Lift the chair, flip it upside down and place it on top of the table.": "Lift the chair, flip it upside down and place it on top of the table.", 
    "Move the chair upside down from the table to the floor.": "Move the chair upside down from the table to the floor.", 
    "Lift the chair, flip it upside down and place it on top of the table. And then move the chair upside down from the table to the floor.": "Lift the chair, flip it upside down and place it on top of the table. And then move the chair upside down from the table to the floor.", 
    "Lift and move the chair. Then kick the chair to move.": "Lift the chair, move the chair, then kick the chair to move, and set it back down.",
}

table_mapping_dict = {
    "Pull the table to the desired location.": "Pull the table, and set it back down.",
    "Lift and move the table to the desired location.": "Lift the table, move the table and put down the table.",
    "Lift the table above your head, spin it and put the table down.": "Lift the table above your head, spin it and put the table down.",
    "Lift the table above your head, walk, and put the table down.": "Lift the table above your head, walk, and put the table down.",
    "Lift the table by one of the edges, rotate it, drag the table and place the legs back down.": "Lift the table by one of the edges, rotate it, drag the table and place the legs back down.",
    "Push to move the table.": "Push the table, and set it back down.",
    "Kick the table to move across the room.": "Kick the table, and set it back down.",
    "Push the table and then drag it to move, then set the table down.": "Push the table, release the hands, then drag the table, and set it back down.", 
    "Lift the table, so only two legs are off the floor. Slide your feet and rotate the table as you slide. Lower the table with your hands.": "Lift the table, so only two legs are off the floor. Slide your feet and rotate the table as you slide. Lower the table with your hands.",
}

box_mapping_dict = {
    "Push and move the box to the desired location.": "Push the box, and set it back down.",
    "Lift the box, carry it a few steps and place it on top of a table.": "Lift the box, move the box, and put down the box.",
    "Pick up the box from the table and put down the box onto the floor.": "Lift the box, move the box, and put down the box.", # Additional one. 
    "Lift the box, carry it toward the desired location, and put it down onto the floor.": "Lift the box, move the box, and put down the box.",
    "Lift, rotate, and place the box back down.": "Lift the box, rotate the box, and set it back down.",
    "Kick the box to slide, lift the box and place it on top of a nearby shelf.": "Kick the box, lift the box, move the box, and put down the box.",
    "Kick the box, then lift the box and move it to the desired location of the floor.": "Kick the box, lift the box, move the box, and put down the box.", # Additional one. For suitcase. 
    "Kick the box to move the box across the room.": "Kick the box, and set it back down.",
    "Pull the box toward you and release the box.": "Pull the box, and set it back down.",
    "Push the box and then pull the the box towards you, causing the box to slide across the floor.": "Push the box, release the hands, then pull the box, and set it back down.",
}
    
monitor_mapping_dict = {
    "Pick up the monitor, walk towards a desk, and put the monitor on the desk.": "Lift the monitor, move the monitor, and put down the monitor.",
    "Pick up the monitor from the desk and put down the monitor onto the floor.": "Lift the monitor, move the monitor, and put down the monitor.", # Additional one. 
    "Pick up the monitor, put the monitor on the desk. Then pick up the monitor from the desk and put down the monitor onto the floor.": "Lift the monitor, move the monitor, and put down the monitor. Then lift the monitor, move the monitor, and put down the monitor.", # Additional one. 
    "Pick up the monitor and move the monitor to the desired location of the floor.": "Lift the monitor, move the monitor, and put down the monitor.", # Additional one. 
    "Lift the monitor, place the monitor on the table, then rotate the monitor to adjust its orientation.": "Lift the monitor, put down the monitor, and then rotate the monitor to adjust its orientation..",
    "Lift the monitor, rotate the monitor and put down the monitor.": "Lift the monitor, rotate the monitor, and put down the monitor.",
    "Grasp the sides of the monitor, tilt it and slide it across a surface.": "Grasp the sides of the monitor, tilt it, pull the monitor, and set it back down.", 
    "Lift the monitor up, walk backward, and place the monitor on the ground.": "Lift the monitor, move the monitor, and put down the monitor.",
    "Pick up the monitor and then rotate the monitor while putting it down.": "Lift the monitor, put down the monitor while rotating it.",
}

floorlamp_mapping_dict = {
    "Lift and move the floorlamp to the desired location.": "Lift the floorlamp, move the floorlamp, and put down the floorlamp.",
    "Pull and move the floorlamp to the desired location.": "Pull the floorlamp, and set it back down.",
    "Kick the base of the floorlamp to move.": "Kick the base of the floorlamp, and set it back down.",
    "Lift and adjust the floorlamp to a different orientation.": "Lift the floorlamp, adjust the orientation of the floorlamp, and set it back down.",
}
    
tripod_mapping_dict = {
    "Pick up and move the tripod to the desired location.": "Lift the tripod, move the tripod, and put down the tripod.",
    "Pull and move the tripod to the desired location.": "Pull the tripod, and set it back down.",
    "Put the tripod down onto the floor, then pick up the tripod.": "Put the tripod horizontally down, then pick up the fallen tripod.",
    "Put the tripod down onto the floor.": "Put the tripod horizontally down.",
    "Pick up the fallen tripod from the floor.": "Pick up the fallen tripod.",
    "Hold and turn the tripod around to a different orientation.": "Hold and turn the tripod around to a different orientation.",
    "Kick the tripod to move.": "Kick the tripod, and set it back down.",
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