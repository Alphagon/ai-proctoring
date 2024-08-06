def alert(condition, no_of_frames):
    if(condition):
        no_of_frames = no_of_frames + 1
    else:
        no_of_frames=0
    return no_of_frames