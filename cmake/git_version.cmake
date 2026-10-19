# Writes OUT with the git commit of SRC (plus "-dirty" if tracked sources
# have uncommitted changes). Run at build time, so every `make` refreshes
# it; the file is rewritten only when its content changes, so an unchanged
# commit does not trigger a recompile.
execute_process(COMMAND git -C ${SRC} rev-parse --short=10 HEAD
                OUTPUT_VARIABLE H OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND git -C ${SRC} status --porcelain --untracked-files=no
                        -- include source examples CMakeLists.txt
                OUTPUT_VARIABLE D OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT H)
  set(H "unknown")
endif()
if(D)
  set(H "${H}-dirty")
endif()
set(C "#define INTERNODES_GIT_VERSION \"${H}\"\n")
if(EXISTS ${OUT})
  file(READ ${OUT} OLD)
endif()
if(NOT "${OLD}" STREQUAL "${C}")
  file(WRITE ${OUT} "${C}")
endif()
