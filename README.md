# lightbox-andy

A lightbox application to rename and delete photos to curate a large collection. Highly optimized to Andy's needs.

## Current Features

- Step through all photos in a folder and show them full-screen.
- Fast navigation: preloads and prescales all photos for instant rendering.
- Simple navigation with arrow keys (left/right) and ESC to exit.
- Loads images from the `photos` directory.
- **Translucent overlay**: Shows the filename and date of the current photo at the bottom of the screen. Toggle the overlay on/off with the F1 key.

## Planned Features

- Ability to order photos by filename or date.
- Ability to filter photos by wildcard/substrings.
- add EXIF info loader, or make an EXIF info extractor to prepare for renaming
- Specialized hot-keys for renaming the shown image according to specific rules.
    - Example format: YYYY MM DD HH:MM Event Location People Description GOOD
    - Optional postfix of DELETE to filename to flag for deletion.
- Functions to get information for the format like Event, Location, People, Description from databases, APIs, file info, etc.
- Carry over "Event", "Location", "Description" between photos for batch editing.
- Flexible input options for setting "Event", "Location", "Description", "People", and "GOOD"/"DELETE" by keyboard.
- Possible feature: Set things using a MIDI keyboard (e.g., Cdim = DELETE, Cmaj = normal, C7 = GOOD).

## Usage

1. Place your photos in the `photos` directory.
2. Run the application. Use the left/right arrow keys to navigate, and ESC to exit.
3. Press **F1** to toggle the overlay with filename and date on or off.

## Development

- The codebase is Java, using Swing for the UI.
- Main classes: `PhotoFinder` (loads and prescales images), `Viewer` (displays and navigates images).
- Contributions and feature ideas are welcome!