import os 
import re

def extract_resolution(path):
    """
    Small function used to obtain the degree level (resolution) from a data file, i.e. '/home/cap003/hall5/weatherbench_paradis/era5_1deg_13level'. 
    """
    # Get the last underscore-delimited part of the basename
    base = os.path.basename(path)
    print('base ', type(base))


    part = next((p for p in base.split("_") if ("deg" in p)), "") # Now find which one of the pieces that we just split has "deg" in it 
    print('part ',part)

    # Use regex to get number before or after "deg" or "degrees"-- re.fullmatch() does not work for this
    match = re.match(r"(?:\d+(?=deg(?:rees)?))|(?:deg(?:rees)?(\d+))", part)
    print('match ',match)
    if match:
        # If number is before "deg", it's the whole match
        num = match.group(0)

        # If number is after "deg", it's in group(1)
        if match.group(1):
            num = match.group(1)

        return num

    return None  # fail gracefully

def extract_version(path):
        """
        Put version number from checkpoint path on the end of output file name   
        """

        match = re.search(r"version_(\d+)", path)
        if match:
            version_num = match.group(1)  
            return version_num
       
        return None
