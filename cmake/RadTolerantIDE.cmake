# RadTolerantIDE.cmake
# Setup for integrated development environments with the radiation-tolerant ML framework

# Include the Cursor configuration
include(${CMAKE_CURRENT_LIST_DIR}/RadTolerantCursorConfig.cmake)

# Function to set up a more advanced clangd configuration for the project
function(configure_clangd_integration)
    set(CLANGD_CONFIG "{
  \"CompileFlags\": {
    \"Add\": [\"-Wall\", \"-Wextra\", \"-std=c++17\", \"-Wno-unused-parameter\"],
    \"Remove\": [\"-W*\", \"-std=*\"]
  },
  \"Diagnostics\": {
    \"UnusedIncludes\": \"Strict\",
    \"ClangTidy\": {
      \"Add\": [\"modernize-*\", \"cppcoreguidelines-*\", \"performance-*\"],
      \"Remove\": [\"modernize-use-trailing-return-type\"],
      \"CheckOptions\": {
        \"cppcoreguidelines-special-member-functions.AllowSoleDefaultDtor\": \"true\"
      }
    }
  },
  \"Index\": {
    \"Background\": \"Build\"
  }
}")

    # Write the clangd configuration file
    file(WRITE "${CMAKE_SOURCE_DIR}/.clangd" "${CLANGD_CONFIG}")
    message(STATUS "clangd configuration created at ${CMAKE_SOURCE_DIR}/.clangd")
endfunction()

# Configure VS Code settings if that IDE is used
function(configure_vscode_integration)
    # Ensure the .vscode directory exists
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/.vscode")
    
    # VS Code settings
    set(VSCODE_SETTINGS "{
  \"editor.formatOnSave\": true,
  \"editor.tabSize\": 4,
  \"editor.insertSpaces\": true,
  \"editor.detectIndentation\": false,
  \"editor.wordWrap\": \"off\",
  \"editor.rulers\": [100],
  \"files.trimTrailingWhitespace\": true,
  \"files.insertFinalNewline\": true,
  \"files.trimFinalNewlines\": true,
  \"files.exclude\": {
    \"**/.git\": true,
    \"**/build\": true,
    \"**/results\": true,
    \"**/CMakeFiles\": true,
    \"**/Testing\": true
  },
  \"files.associations\": {
    \"*.hpp\": \"cpp\",
    \"*.cpp\": \"cpp\",
    \"*.h\": \"cpp\",
    \"*.c\": \"cpp\",
    \"*.tpp\": \"cpp\",
    \"CMakeLists.txt\": \"cmake\"
  },
  \"C_Cpp.default.cppStandard\": \"c++17\",
  \"C_Cpp.default.includePath\": [
    \"${CMAKE_SOURCE_DIR}/include\",
    \"${CMAKE_SOURCE_DIR}/src\"
  ],
  \"C_Cpp.default.defines\": [
    \"RAD_ML_DEBUG=${CMAKE_BUILD_TYPE}\",
    \"RAD_ML_VERSION=\\\"${CMAKE_PROJECT_VERSION}\\\"\"
  ],
  \"C_Cpp.clang_format_style\": \"{ BasedOnStyle: Google, IndentWidth: 4, ColumnLimit: 100, NamespaceIndentation: None, BreakBeforeBraces: Stroustrup }\",
  \"cmake.configureOnOpen\": true,
  \"cmake.buildDirectory\": \"${CMAKE_SOURCE_DIR}/build\"
}")

    # VS Code extensions recommendations
    set(VSCODE_EXTENSIONS "{
  \"recommendations\": [
    \"ms-vscode.cpptools\",
    \"ms-vscode.cmake-tools\",
    \"xaver.clang-format\",
    \"twxs.cmake\",
    \"cschlosser.doxdocgen\",
    \"ms-python.python\"
  ]
}")

    # Write VS Code configuration files
    file(WRITE "${CMAKE_SOURCE_DIR}/.vscode/settings.json" "${VSCODE_SETTINGS}")
    file(WRITE "${CMAKE_SOURCE_DIR}/.vscode/extensions.json" "${VSCODE_EXTENSIONS}")
    
    message(STATUS "VS Code integration configured")
endfunction()

# Configure doxygen settings for the project
function(configure_doxygen_integration)
    # Create initial Doxyfile if it doesn't exist
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/docs/Doxyfile")
        file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/docs")
        
        set(DOXYGEN_CONFIG "
PROJECT_NAME           = \"Radiation Tolerant ML Framework\"
PROJECT_NUMBER         = ${CMAKE_PROJECT_VERSION}
PROJECT_BRIEF          = \"ML framework with radiation-tolerant features for space applications\"
OUTPUT_DIRECTORY       = ${CMAKE_SOURCE_DIR}/docs/generated
INPUT                  = ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src
RECURSIVE              = YES
EXTRACT_ALL            = YES
EXTRACT_PRIVATE        = YES
EXTRACT_PACKAGE        = YES
EXTRACT_STATIC         = YES
EXTRACT_LOCAL_CLASSES  = YES
FILE_PATTERNS          = *.cpp *.hpp *.h *.c *.tpp
GENERATE_HTML          = YES
GENERATE_LATEX         = NO
USE_MDFILE_AS_MAINPAGE = ${CMAKE_SOURCE_DIR}/README.md
HTML_EXTRA_STYLESHEET  = 
HTML_COLORSTYLE_HUE    = 220
HTML_COLORSTYLE_SAT    = 100
HTML_COLORSTYLE_GAMMA  = 80
HTML_DYNAMIC_SECTIONS  = YES
HAVE_DOT               = YES
UML_LOOK               = YES
CALL_GRAPH             = YES
CALLER_GRAPH           = YES
DOT_IMAGE_FORMAT       = svg
INTERACTIVE_SVG        = YES
PLANTUML_JAR_PATH      = 
PLANTUML_CFG_FILE      = 
")
        
        file(WRITE "${CMAKE_SOURCE_DIR}/docs/Doxyfile" "${DOXYGEN_CONFIG}")
        message(STATUS "Doxygen configuration created at ${CMAKE_SOURCE_DIR}/docs/Doxyfile")
    endif()
    
    # Add doxygen target if doxygen is found
    find_package(Doxygen QUIET)
    if(DOXYGEN_FOUND)
        add_custom_target(
            docs
            ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/docs/Doxyfile
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
        message(STATUS "Doxygen target 'docs' added")
    endif()
endfunction()

# Configure all IDE integrations
function(configure_all_ide_integrations)
    # Configure cursor settings
    configure_rad_tolerant_cursor()
    
    # Configure clangd integration
    configure_clangd_integration()
    
    # Configure VS Code (if used)
    configure_vscode_integration()
    
    # Configure doxygen integration
    configure_doxygen_integration()
    
    # Add a combined target to update all IDE configurations
    add_custom_target(
        update_ide_settings
        DEPENDS update_cursor_settings
        COMMAND ${CMAKE_COMMAND} -E echo "All IDE settings updated"
        COMMENT "Updating all IDE integrations for rad-tolerant-ml"
    )
    
    message(STATUS "IDE integrations configured successfully")
endfunction() 