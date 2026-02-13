# pybind11 is pulled in by the top-level ORB_SLAM3_pybind CMake via
# `Thirdparty/pybind11/pybind11.cmake` (FetchContent), so we don't require a
# system-wide pybind11Config.cmake here.
pybind11_add_module(orb_slam_pybind MODULE orb_slam3_pybind.cpp) 
target_link_libraries(orb_slam_pybind PUBLIC ORB_SLAM3 g2o DBoW2)
set(_PYMOD_OUT "${PROJECT_SOURCE_DIR}/python_wrapper")
set_target_properties(orb_slam_pybind PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_PYMOD_OUT}"
    RUNTIME_OUTPUT_DIRECTORY "${_PYMOD_OUT}"
    ARCHIVE_OUTPUT_DIRECTORY "${_PYMOD_OUT}"
)
# Multi-config generators (e.g. Visual Studio) otherwise place binaries under a config subdir.
foreach(_cfg Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${_cfg}" _CFG_UP)
    set_target_properties(orb_slam_pybind PROPERTIES
        "LIBRARY_OUTPUT_DIRECTORY_${_CFG_UP}" "${_PYMOD_OUT}"
        "RUNTIME_OUTPUT_DIRECTORY_${_CFG_UP}" "${_PYMOD_OUT}"
        "ARCHIVE_OUTPUT_DIRECTORY_${_CFG_UP}" "${_PYMOD_OUT}"
    )
endforeach()
install(TARGETS orb_slam_pybind DESTINATION .)
