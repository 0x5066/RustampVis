# Winamp Classic Mini Visualizer

This project supercedes [PynampVis](https://github.com/0x5066/PynampVis) and development on that has stopped since this version was being worked on.

RustampVis comes with a configurable UI that's made to look like Skinned Preferences from the early Winamp and early WACUP days, this program also supports loading of any gen_ex.png file.

This project retains the support for loading and applying a custom viscolor.txt, but also supports images that're in the viscolor.txt compliant format (both normal and 16osc versions) in either the X or Y positions.

Support for Winamp-style configs, meaning you can apply a ``winamp.ini`` file through ``-c winamp.ini`` to load the expected visualizer settings and have them behave mostly the same (support for the FPS limiter does not exist, however).

Some extra entries also exist that either only exist for RustampVis (``sa_amp`` and ``prefs_id``) or WACUP.

Command line usage:
```
Options:
  -v, --viscolor <VISCOLOR>    Name of the custom viscolor.txt file, supports images in the viscolor.txt format as well [default: viscolor.txt]
  -d, --device <DEVICE>        Index of the audio device to use
  -z, --zoom <ZOOM>            Zoom factor [default: 7]
  -c, --configini <CONFIGINI>  Name of the config file [default: rustampvis.ini]
      --debug                  Debug
  -h, --help                   Print help
  -V, --version                Print version
```

Supported OSes:

Windows (*7 to 11) and Linux are fully supported, for macOS it does compile, but upon selecting the output device the program just freezes with no error.

You *will* have to install SDL2, SDL2_image and SDL2_ttf for the program to work correctly.


*Windows 7 is supported, but the WASAPI loopback capture isn't working for some reason.