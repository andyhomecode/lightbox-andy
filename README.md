# lightbox-andy
A lightbox application to rename and delete photos to curate a large collection.  Highly optimized to Andy's needs

## Function

- Step through all photos in a folder and show them full-screen

- speed of navigation is key, so pre-fetch photos to make rendering as fast as possible.

- have a translucent overlay to show key information like filename and date, and other data that may be pulled from a database in the future.

- Ability to order photos by filename, date

- Ability to filter photos by wildcard/substrings

- Specalized hot-keys for renaming the shown image according to specific rules

- Example format: YYYY MM DD HH:MM Event Location People Description GOOD

- optional postfix of DELETE to filename to flag for deletion.

- Functions to get information for the format like Event Location, People, Description from databases, APIs, file info, etc.

- "Event", "Location", "Description" will be carried between photos to quickly apply to may photos in a sequence.

- Flexible input options for setting "Event" "Location" "Description" "People" and "GOOD" or "DELETE" by keyboard

- Possible feature: Set things using a MIDI keyboard.  Example:  Cdim= DELETE.  Cmaj: normal settings. C7 GOOD.  Need to think through, but leave space for chorded input. settings.