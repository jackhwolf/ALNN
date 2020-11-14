#!/bin/bash

if [ $(hostname) != SkippyElvis ];
    then
        module load ffmpeg-4.3.1
fi

pushd $1
# frames --> video with 0.5 frames/second == 1 frame/2 seconds
# defautl framerrate is 25fps, so filter by 25*2=50 --> fps = 25/50 = 0.5
ffmpeg -i %05d.png -filter:v "setpts=50.0*PTS" -vcodec libx264 $2
popd