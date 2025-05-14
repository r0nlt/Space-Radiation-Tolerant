# RadTolerantCursorConfig.cmake
# CMake integration with Cursor IDE for the radiation-tolerant ML framework

# Create .cursor.json in the project root
function(configure_cursor_settings)
    # Define the cursor configuration content (without comments to be valid JSON)
    set(CURSOR_CONFIG "{
  \"editor\": {
    \"formatOnSave\": true,
    \"tabSize\": 4,
    \"insertSpaces\": true,
    \"detectIndentation\": false,
    \"wordWrap\": \"off\",
    \"rulers\": [100]
  },
  \"clangd\": {
    \"enableInlayHints\": true,
    \"checkUpdates\": true
  },
  \"autocomplete\": {
    \"enabled\": true,
    \"delay\": 0
  },
  \"typescript\": {
    \"format\": {
      \"indentSize\": 4,
      \"semicolons\": \"insert\",
      \"trailingCommas\": \"all\"
    }
  },
  \"python\": {
    \"format\": {
      \"indentSize\": 4,
      \"lineLength\": 100
    },
    \"linter\": \"pyright\"
  },
  \"cpp\": {
    \"format\": {
      \"indentWidth\": 4,
      \"columnLimit\": 100,
      \"useTab\": false,
      \"namespaceIndentation\": \"None\",
      \"accessModifierOffset\": -4,
      \"breakBeforeBraces\": \"Stroustrup\",
      \"pointerAlignment\": \"Right\",
      \"includeBlocks\": \"Preserve\",
      \"breakConstructorInitializers\": \"BeforeColon\"
    },
    \"intelliSense\": {
      \"mode\": \"clang-x64\"
    }
  },
  \"files\": {
    \"trimTrailingWhitespace\": true,
    \"insertFinalNewline\": true,
    \"trimFinalNewlines\": true,
    \"exclude\": {
      \"**/.git\": true,
      \"**/build\": true,
      \"**/results\": true,
      \"**/CMakeFiles\": true,
      \"**/Testing\": true,
      \"**/third_party\": false
    },
    \"associations\": {
      \"*.hpp\": \"cpp\",
      \"*.cpp\": \"cpp\",
      \"*.h\": \"cpp\",
      \"*.c\": \"cpp\",
      \"*.tpp\": \"cpp\",
      \"CMakeLists.txt\": \"cmake\"
    }
  },
  \"terminal\": {
    \"fontFamily\": \"monospace\",
    \"fontSize\": 14,
    \"lineHeight\": 1.2
  },
  \"vim\": {
    \"enabled\": false
  },
  \"git\": {
    \"enableSmartCommit\": true,
    \"autofetch\": true
  }
}")

    # Write the configuration to .cursor.json in the project root
    file(WRITE "${CMAKE_SOURCE_DIR}/.cursor.json" "${CURSOR_CONFIG}")
    message(STATUS "Cursor IDE configuration created at ${CMAKE_SOURCE_DIR}/.cursor.json")
endfunction()

# Create a compile_commands.json for better code intelligence
function(configure_cursor_compile_commands)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON PARENT_SCOPE)
    message(STATUS "Enabled compilation database for Cursor IDE")
endfunction()

# Configure Cursor to highlight radiation-tolerant patterns
function(configure_cursor_custom_highlighting)
    set(HIGHLIGHTING_CONFIG "{
  \"highlight.regexpHighlight\": [
    {
      \"pattern\": \"TMR<[^>]+>|EnhancedTMR<[^>]+>\",
      \"color\": \"#73D0FF\",
      \"backgroundColor\": \"#3a414d\"
    },
    {
      \"pattern\": \"::memory::|::scrubber::\",
      \"color\": \"#7AF8CA\",
      \"backgroundColor\": \"#3a414d\"
    },
    {
      \"pattern\": \"rad_ml::(quantum|physics)::\",
      \"color\": \"#FF9E64\",
      \"backgroundColor\": \"#3a414d\"
    },
    {
      \"pattern\": \"MissionProfile\",
      \"color\": \"#D4BFFF\",
      \"backgroundColor\": \"#3a414d\"
    }
  ]
}")

    # Write the custom highlighting configuration
    file(WRITE "${CMAKE_SOURCE_DIR}/.cursor-highlighting.json" "${HIGHLIGHTING_CONFIG}")
    message(STATUS "Custom radiation-tolerant pattern highlighting configured")
endfunction()

# Configure project initialization
function(configure_rad_tolerant_cursor)
    configure_cursor_settings()
    configure_cursor_compile_commands()
    configure_cursor_custom_highlighting()
    
    # Create a convenience target
    add_custom_target(
        update_cursor_settings
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_LIST_FILE}
        COMMENT "Updating Cursor IDE settings for rad-tolerant-ml"
    )
endfunction()

# If this script is run directly via cmake -P, execute the configuration
if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
    configure_rad_tolerant_cursor()
endif() 