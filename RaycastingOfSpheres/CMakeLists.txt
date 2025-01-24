set(PROJECT_NAME RaycastingOfSpheres)

################################################################################
# Source groups
################################################################################
set(no_group_source_files
    "random_objects_generator.h"
)
source_group("" FILES ${no_group_source_files})

set(Headers
    "camera.h"
    "float3Operators.h"
    "kernel.h"
    "lightSource.h"
    "linmath.h"
    "sphere.h"
)
source_group("Headers" FILES ${Headers})

set(Sources
    "../includes/imgui/imgui.cpp"
    "../includes/imgui/imgui_demo.cpp"
    "../includes/imgui/imgui_draw.cpp"
    "../includes/imgui/imgui_impl_glfw.cpp"
    "../includes/imgui/imgui_impl_opengl3.cpp"
    "../includes/imgui/imgui_tables.cpp"
    "../includes/imgui/imgui_widgets.cpp"
    "glad.c"
    "lightSource.cpp"
    "main.cu"
    "random_objects_generator.cpp"
    "sphere.cu"
)
source_group("Sources" FILES ${Sources})

set(ALL_FILES
    ${no_group_source_files}
    ${Headers}
    ${Sources}
)

################################################################################
# Target
################################################################################
add_executable(${PROJECT_NAME} ${ALL_FILES})

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
################################################################################
# Includes for CMake from *.props
################################################################################
use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "$ENV{VCTargetsPath}/BuildCustomizations/CUDA 12.6.cmake")

set(ROOT_NAMESPACE RaycastingOfSpheres)

set_target_properties(${PROJECT_NAME} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
)
################################################################################
# Compile definitions
################################################################################
target_compile_definitions(${PROJECT_NAME} PRIVATE
    "$<$<CONFIG:Debug>:"
        "_DEBUG"
    ">"
    "$<$<CONFIG:Release>:"
        "NDEBUG"
    ">"
    "WIN32;"
    "WIN64;"
    "_CONSOLE;"
    "_MBCS"
)

################################################################################
# Compile and link options
################################################################################
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:
            /Od
        >
        $<$<CONFIG:Release>:
            /O2;
            /Oi;
            /Gy
        >
        /W3;
        ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
        ${DEFAULT_CXX_EXCEPTION_HANDLING}
    )
    target_link_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:
            /INCREMENTAL
        >
        $<$<CONFIG:Release>:
            /OPT:REF;
            /OPT:ICF
        >
        /DEBUG;
        /SUBSYSTEM:CONSOLE
    )
endif()

################################################################################
# Dependencies
################################################################################
set(ADDITIONAL_LIBRARY_DEPENDENCIES
    "glfw3;"
    "opengl32;"
    "cudart_static;"
    "kernel32;"
    "user32;"
    "gdi32;"
    "winspool;"
    "comdlg32;"
    "advapi32;"
    "shell32;"
    "ole32;"
    "oleaut32;"
    "uuid;"
    "odbc32;"
    "odbccp32"
)
target_link_libraries(${PROJECT_NAME} PRIVATE "${ADDITIONAL_LIBRARY_DEPENDENCIES}")

