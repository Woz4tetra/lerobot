# Shared camera configuration for SO-101 follower.
# Sourced by the top-level scripts so all --robot.cameras settings stay in sync.
# Edit camera indices, resolution, fps, and v4l2 controls here in one place.

export ROBOT_CAMERAS="{ \
  gripper: { \
    type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 31, backend: V4L2, \
    v4l2_controls: { \
      auto_exposure: 1, exposure_time_absolute: 157, exposure_dynamic_framerate: 0, \
      white_balance_automatic: 0, white_balance_temperature: 3830, hue: 15 \
    } \
  }, \
  overhead: { \
    type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2, \
    v4l2_controls: { \
      auto_exposure: 1, exposure_time_absolute: 150, exposure_dynamic_framerate: 0, \
      white_balance_automatic: 0, white_balance_temperature: 3669, focus_automatic_continuous: 0 \
    } \
  } \
}"
