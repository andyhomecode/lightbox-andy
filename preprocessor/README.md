# This is the preprocessor to compile info about each photo to make categorization faster

Written in Python for ease of development, AI, and library use.

## this is currently medum level garbage.

- face detection isn't good. tried face_recognition and it was terrible.
- Insight face is better but not good by any means.
- maybe need to abandon face recognition for now, focus on dupes & geo.



## Features:


* finds duplicate/near-duplicate images
* ranks the “best” in each group from criteria such as sharpness, contract, includes faces, composition 
* identifies people "Andy", "Michele", "Person1" 
* extracts EXIF (including GPS as decimal lat/lon, camera info, timestamp),
* optionally tags each photo with a human label (e.g., **“Birthday party”**, **“Beach”**, **“Boating”**, **“Camping”**) using CLIP zero-shot,
* writes a clean **CSV catalog** + a **JSON** of dup groups, and
