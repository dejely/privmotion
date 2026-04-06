set(the_description "Privacy-preserving kinematic analytics")

ocv_define_module(privmotion
    opencv_core
    OPTIONAL opencv_imgproc opencv_videoio
    WRAP python
)

